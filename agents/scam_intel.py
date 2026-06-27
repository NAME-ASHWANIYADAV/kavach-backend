"""
KAVACH 2.0 — Agent 1: Scam Intelligence
=========================================
The first agent in the KAVACH pipeline. Analyzes user-submitted text
(scam messages, call transcripts, WhatsApp forwards) and produces:

1. Risk Score (0.0 - 1.0, displayed as 0-100%)
2. Scam Type Classification
3. Extracted Entities (phone numbers, UPI IDs, account numbers, amounts)
4. RAG Citations from official MHA/RBI/I4C advisories

Design:
    - Entity extraction uses regex + LLM for hybrid accuracy
    - Risk scoring uses a weighted combination of:
        a) Pattern matching against known scam templates
        b) RAG similarity to government advisories
        c) LLM-based semantic analysis
    - Citations are ONLY from verified government sources (no hallucination)
"""

from __future__ import annotations

import re
import json
import logging
from typing import Optional
from pathlib import Path

import google.generativeai as genai

from config import settings
from rag.retriever import retrieve_advisories


logger = logging.getLogger("kavach.scam_intel")


# ─── Known Scam Patterns ────────────────────────────────────────────
# High-confidence regex patterns for common Indian scam scripts.
# These provide deterministic detection even without LLM availability.

SCAM_PATTERNS = {
    "digital_arrest": {
        "weight": 0.35,
        "patterns": [
            r"(?i)(cbi|police|customs|narcotics|enforcement|cyber\s*cell|crime\s*branch)",
            r"(?i)(arrest\s*warrant|fir\s*filed|case\s*registered|under\s*investigation)",
            r"(?i)(aadhaar|aadhar|pan\s*card|passport).*?(linked|used|involved|misused)",
            r"(?i)(money\s*laundering|drug\s*trafficking|hawala|terror\s*funding)",
            r"(?i)(transfer|send|deposit|pay).*?(verification|security|clearance)",
            r"(?i)(do\s*not\s*tell|don.?t\s*inform|keep\s*secret|confidential\s*matter)",
            r"(?i)(supreme\s*court|high\s*court|court\s*order|judicial\s*order)",
            r"(?i)(digital\s*arrest|house\s*arrest|virtual\s*custody)",
        ],
    },
    "upi_fraud": {
        "weight": 0.30,
        "patterns": [
            r"(?i)(upi|paytm|phonepe|gpay|google\s*pay|bhim).*?(refund|cashback|reward)",
            r"(?i)(kyc|update|verify|expire).*?(account|bank|upi)",
            r"(?i)(click|tap|open).*?(link|url|http)",
            r"(?i)(otp|pin|password|cvv).*?(share|enter|provide|send)",
        ],
    },
    "impersonation": {
        "weight": 0.25,
        "patterns": [
            r"(?i)(i\s*am|this\s*is|speaking\s*from).*?(officer|inspector|director|commissioner)",
            r"(?i)(rbi|reserve\s*bank|sebi|income\s*tax|trai|telecom)",
            r"(?i)(your\s*number|your\s*account|your\s*sim|your\s*aadhaar).*?(blocked|suspended|deactivated)",
        ],
    },
    "lottery_investment": {
        "weight": 0.10,
        "patterns": [
            r"(?i)(lottery|lucky\s*draw|prize|winner|jackpot|congrat)",
            r"(?i)(investment|guaranteed\s*return|double\s*money|100%\s*profit)",
            r"(?i)(bitcoin|crypto|forex|trading\s*opportunity)",
        ],
    },
}


# ─── Entity Extraction ──────────────────────────────────────────────

ENTITY_REGEXES = {
    "phone": r"(?:\+91[\s-]?)?(?:[6-9]\d{9})",
    "upi_id": r"[a-zA-Z0-9._-]+@[a-zA-Z]{2,}",
    "account_number": r"\b\d{9,18}\b",
    "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "amount": r"(?:₹|rs\.?|inr)\s*[\d,]+(?:\.\d{2})?",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
}


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from scam text using regex patterns."""
    entities = []
    seen = set()

    for entity_type, pattern in ENTITY_REGEXES.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                entities.append({
                    "entity_type": entity_type,
                    "value": cleaned,
                    "confidence": 0.85,
                    "source_text": text[max(0, text.find(cleaned) - 20):text.find(cleaned) + len(cleaned) + 20],
                })

    return entities


# ─── Pattern-Based Risk Scoring ─────────────────────────────────────

def compute_pattern_score(text: str) -> tuple[float, str]:
    """
    Compute a risk score based on regex pattern matching.
    Returns (score, detected_scam_type).
    """
    best_score = 0.0
    best_type = "unknown"

    for scam_type, config in SCAM_PATTERNS.items():
        matches = 0
        total = len(config["patterns"])

        for pattern in config["patterns"]:
            if re.search(pattern, text):
                matches += 1

        if total > 0:
            match_ratio = matches / total
            weighted_score = match_ratio * config["weight"]

            if weighted_score > best_score:
                best_score = weighted_score
                best_type = scam_type

    # Normalize: if many patterns match, boost the score significantly
    return min(best_score * 3.0, 1.0), best_type


# ─── LLM-Based Analysis ────────────────────────────────────────────

ANALYSIS_PROMPT = """You are KAVACH, an expert AI system for detecting digital fraud scams in India.

Analyze the following message/transcript and provide a structured assessment.

INPUT MESSAGE:
\"\"\"
{user_input}
\"\"\"

RELEVANT GOVERNMENT ADVISORIES (from MHA/RBI/I4C):
{advisory_context}

TASK:
1. Determine if this is a scam. Classify the type: "digital_arrest", "upi_fraud", "impersonation", "lottery_investment", "phishing", or "legitimate".
2. Provide a risk score from 0.0 (safe) to 1.0 (definite scam).
3. Write a brief analysis (2-3 sentences) explaining why this is or isn't a scam.
4. Extract any entities not already captured: phone numbers, UPI IDs, account numbers, amounts, names.

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no code blocks):
{{
    "risk_score": 0.0,
    "scam_type": "type_here",
    "analysis": "Your analysis here",
    "additional_entities": [
        {{"entity_type": "phone", "value": "9876543210", "confidence": 0.9}}
    ]
}}"""


def llm_analyze(user_input: str, advisory_context: str) -> dict:
    """Use Gemini to perform semantic scam analysis."""
    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        prompt = ANALYSIS_PROMPT.format(
            user_input=user_input,
            advisory_context=advisory_context,
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=1024,
            ),
        )

        # Parse JSON from response
        response_text = response.text.strip()
        # Clean markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r"```(?:json)?\n?", "", response_text).strip()

        result = json.loads(response_text)
        return {
            "risk_score": float(result.get("risk_score", 0.0)),
            "scam_type": result.get("scam_type", "unknown"),
            "analysis": result.get("analysis", ""),
            "additional_entities": result.get("additional_entities", []),
        }

    except Exception as e:
        logger.warning(f"LLM analysis failed, using pattern-only: {e}")
        return None


# ─── Main Analysis Function ─────────────────────────────────────────

def analyze_scam_message(user_input: str) -> dict:
    """
    Full scam analysis pipeline:
    1. Pattern-based scoring (deterministic, fast)
    2. Entity extraction (regex)
    3. RAG retrieval (advisory citations)
    4. LLM analysis (semantic understanding)
    5. Combine scores with weighted averaging

    Returns dict with: risk_score, risk_level, scam_type, scam_analysis,
                       extracted_entities, rag_citations
    """
    logger.info(f"Analyzing message: {user_input[:80]}...")

    # Step 1: Pattern-based analysis
    pattern_score, pattern_type = compute_pattern_score(user_input)

    # Step 2: Entity extraction
    entities = extract_entities(user_input)

    # Step 3: RAG retrieval
    rag_results = retrieve_advisories(user_input)
    citations = [
        {
            "source_document": r["source"],
            "relevant_text": r["text"],
            "similarity_score": r["score"],
            "advisory_date": r.get("date", ""),
            "issuing_authority": r.get("authority", "I4C"),
        }
        for r in rag_results
    ]

    # Compute RAG boost: if advisories are very similar, boost score
    rag_boost = 0.0
    if citations:
        max_similarity = max(c["similarity_score"] for c in citations)
        rag_boost = max_similarity * 0.3  # Up to 0.3 boost from RAG

    # Step 4: LLM analysis
    advisory_context = "\n\n".join(
        f"[{c['source_document']}]: {c['relevant_text'][:300]}"
        for c in citations[:3]
    ) if citations else "No matching advisories found."

    llm_result = llm_analyze(user_input, advisory_context)

    # Step 5: Combine scores
    if llm_result:
        # Weighted average: 40% pattern + 30% LLM + 30% RAG
        combined_score = (
            pattern_score * 0.40
            + llm_result["risk_score"] * 0.30
            + rag_boost
        )
        scam_type = llm_result["scam_type"] if llm_result["risk_score"] > 0.4 else pattern_type
        analysis = llm_result["analysis"]

        # Merge LLM-extracted entities
        for ent in llm_result.get("additional_entities", []):
            if isinstance(ent, dict) and ent.get("value"):
                existing_values = {e["value"] for e in entities}
                if ent["value"] not in existing_values:
                    entities.append({
                        "entity_type": ent.get("entity_type", "unknown"),
                        "value": ent["value"],
                        "confidence": ent.get("confidence", 0.7),
                        "source_text": "",
                    })
    else:
        # Fallback: pattern + RAG only
        combined_score = pattern_score + rag_boost
        scam_type = pattern_type
        analysis = f"Pattern analysis detected {scam_type} indicators."

    # Clamp score
    combined_score = max(0.0, min(1.0, combined_score))

    # Determine risk level
    if combined_score >= 0.85:
        risk_level = "CRITICAL"
    elif combined_score >= 0.65:
        risk_level = "HIGH"
    elif combined_score >= 0.40:
        risk_level = "MEDIUM"
    elif combined_score >= 0.15:
        risk_level = "LOW"
    else:
        risk_level = "SAFE"

    result = {
        "risk_score": combined_score,
        "risk_level": risk_level,
        "scam_type": scam_type,
        "scam_analysis": analysis,
        "extracted_entities": entities,
        "rag_citations": citations,
    }

    logger.info(f"Analysis complete: {risk_level} ({combined_score:.2%}) - {scam_type}")
    return result
