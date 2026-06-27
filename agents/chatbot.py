"""
KAVACH 2.0 — Agent 4: Citizen Triage Chatbot
==============================================
A deterministic 5-step state machine for victim triage.
Uses a structured flow to collect complaint information,
NOT free-form LLM chat (to prevent hallucinations).

Gemini is used ONLY for:
    1. Final risk summary generation
    2. Hindi/English language handling

Design:
    Step 0: Greeting + language detection
    Step 1: Scam type identification
    Step 2: Incident details collection
    Step 3: Entity extraction + verification
    Step 4: Risk assessment + action plan
    Step 5: Generate complaint packet + freeze order trigger

Why deterministic:
    - Legally auditable conversation flow
    - No hallucinated advice to victims
    - Consistent data collection for CFCFRMS
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import google.generativeai as genai

from config import settings


logger = logging.getLogger("kavach.chatbot")


# ─── Triage Steps ───────────────────────────────────────────────────

TRIAGE_STEPS = {
    0: {
        "name": "greeting",
        "message_en": (
            "🛡️ **KAVACH — Digital Safety Shield**\n\n"
            "Namaste! I'm here to help you if you've received a suspicious call, "
            "message, or are being threatened online.\n\n"
            "**Tell me: What happened?** You can describe it in Hindi or English.\n\n"
            "Or choose:\n"
            "1️⃣ I received a suspicious call/message\n"
            "2️⃣ I was threatened with 'digital arrest'\n"
            "3️⃣ I already transferred money to a scammer\n"
            "4️⃣ I want to check if a message is a scam"
        ),
        "message_hi": (
            "🛡️ **KAVACH — डिजिटल सुरक्षा कवच**\n\n"
            "नमस्ते! मैं आपकी मदद के लिए यहाँ हूँ। अगर आपको कोई संदिग्ध कॉल, "
            "मैसेज आया है या ऑनलाइन धमकी दी जा रही है।\n\n"
            "**बताइए: क्या हुआ?** आप हिंदी या English में बता सकते हैं।\n\n"
            "या चुनें:\n"
            "1️⃣ मुझे संदिग्ध कॉल/मैसेज आया\n"
            "2️⃣ 'डिजिटल अरेस्ट' की धमकी दी गई\n"
            "3️⃣ मैंने पहले ही पैसे ट्रांसफर कर दिए\n"
            "4️⃣ मैं चेक करना चाहता हूँ कि कोई मैसेज स्कैम है"
        ),
    },
    1: {
        "name": "scam_type",
        "message_en": (
            "I understand. Let me help you assess this situation.\n\n"
            "**What type of contact was it?**\n"
            "1️⃣ Video call (Skype/WhatsApp) — someone claimed to be police/CBI\n"
            "2️⃣ Phone call — threatening arrest or legal action\n"
            "3️⃣ SMS/WhatsApp message — with a link or OTP request\n"
            "4️⃣ Email — claiming your account is blocked\n"
            "5️⃣ Other"
        ),
        "message_hi": (
            "समझ गया। मैं आपकी मदद करता हूँ।\n\n"
            "**किस तरह का संपर्क था?**\n"
            "1️⃣ वीडियो कॉल (Skype/WhatsApp) — पुलिस/CBI बनकर\n"
            "2️⃣ फोन कॉल — गिरफ्तारी या कानूनी कार्रवाई की धमकी\n"
            "3️⃣ SMS/WhatsApp मैसेज — लिंक या OTP मांगा\n"
            "4️⃣ ईमेल — अकाउंट ब्लॉक होने की बात\n"
            "5️⃣ कुछ और"
        ),
    },
    2: {
        "name": "details",
        "message_en": (
            "Thank you. Now I need some details:\n\n"
            "**Please share any of the following (whatever you have):**\n"
            "• The phone number that called/messaged you\n"
            "• Any UPI ID they asked you to send money to\n"
            "• The amount they demanded or you transferred\n"
            "• A screenshot or copy of the message\n\n"
            "💡 *Don't worry — this information stays private and is only used for analysis.*"
        ),
        "message_hi": (
            "धन्यवाद। अब मुझे कुछ जानकारी चाहिए:\n\n"
            "**कृपया ये बताएं (जो भी हो):**\n"
            "• जिस नंबर से कॉल/मैसेज आया\n"
            "• कोई UPI ID जिस पर पैसे भेजने को कहा\n"
            "• कितने पैसे मांगे या भेजे\n"
            "• मैसेज का स्क्रीनशॉट या कॉपी\n\n"
            "💡 *चिंता न करें — आपकी जानकारी सुरक्षित है और सिर्फ विश्लेषण के लिए है।*"
        ),
    },
    3: {
        "name": "confirmation",
        "message_en": (
            "Got it. Let me analyze this information...\n\n"
            "🔍 **Running KAVACH Intelligence Pipeline:**\n"
            "• Scam pattern analysis ✅\n"
            "• Entity extraction ✅\n"
            "• Fraud network check ⏳\n\n"
            "*Processing...*"
        ),
        "message_hi": (
            "समझ गया। जानकारी का विश्लेषण कर रहा हूँ...\n\n"
            "🔍 **KAVACH इंटेलिजेंस पाइपलाइन चल रही है:**\n"
            "• स्कैम पैटर्न विश्लेषण ✅\n"
            "• एंटिटी एक्सट्रैक्शन ✅\n"
            "• फ्रॉड नेटवर्क चेक ⏳\n\n"
            "*प्रोसेसिंग...*"
        ),
    },
    4: {
        "name": "action_plan",
        "message_en": (
            "⚠️ **KAVACH ALERT — Action Required**\n\n"
            "{risk_assessment}\n\n"
            "**Immediate Steps:**\n"
            "1. 🚨 **Call 1930** (Cyber Crime Helpline) — Report immediately\n"
            "2. 🏦 **Contact your bank** — Request transaction reversal\n"
            "3. 📝 **File online complaint** at cybercrime.gov.in\n"
            "4. 📸 **Preserve evidence** — Don't delete messages/call logs\n\n"
            "{freeze_info}\n\n"
            "**Remember:** No government agency conducts arrests via video calls. "
            "You are NOT in trouble — the caller was a criminal."
        ),
        "message_hi": (
            "⚠️ **KAVACH अलर्ट — तुरंत कार्रवाई करें**\n\n"
            "{risk_assessment}\n\n"
            "**तुरंत करें:**\n"
            "1. 🚨 **1930 पर कॉल करें** (साइबर क्राइम हेल्पलाइन)\n"
            "2. 🏦 **अपने बैंक से संपर्क करें** — ट्रांजैक्शन रिवर्सल\n"
            "3. 📝 **ऑनलाइन शिकायत** cybercrime.gov.in पर दर्ज करें\n"
            "4. 📸 **सबूत सुरक्षित रखें** — मैसेज/कॉल लॉग न मिटाएं\n\n"
            "{freeze_info}\n\n"
            "**याद रखें:** कोई भी सरकारी एजेंसी वीडियो कॉल से गिरफ्तार नहीं करती। "
            "आप सुरक्षित हैं — कॉल करने वाला अपराधी था।"
        ),
    },
}


def _detect_language(text: str) -> str:
    """Simple Hindi detection based on Devanagari script presence."""
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return "hi" if devanagari_count > len(text) * 0.15 else "en"


def _generate_risk_summary(triage_data: dict, risk_score: float = 0.0) -> str:
    """Generate a brief risk assessment summary."""
    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_FLASH_MODEL)

        prompt = f"""Based on this fraud complaint data, generate a brief 2-sentence risk assessment in simple language:
Scam Type: {triage_data.get('scam_type', 'Unknown')}
Details: {triage_data.get('details', 'Not provided')}
Risk Score: {risk_score:.0%}

Format: "Risk Level: HIGH/MEDIUM/LOW. [Explanation]"
Keep it under 50 words. Be direct and helpful."""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=150),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        if risk_score >= 0.65:
            return "Risk Level: HIGH. This shows strong indicators of a digital arrest scam. Take action immediately."
        elif risk_score >= 0.40:
            return "Risk Level: MEDIUM. This has suspicious patterns. Stay cautious and verify through official channels."
        else:
            return "Risk Level: LOW. This may be legitimate, but always verify through official sources."


def process_chat_message(
    user_message: str,
    chat_history: list[dict],
    triage_step: int,
    triage_data: dict,
    risk_score: float = 0.0,
) -> dict:
    """
    Process a chat message through the triage state machine.

    Args:
        user_message: User's current message
        chat_history: Previous messages
        triage_step: Current step (0-5)
        triage_data: Collected triage information
        risk_score: Risk score from pipeline (if available)

    Returns:
        dict with: response, triage_step, triage_data, chat_history,
                   should_trigger_pipeline (bool)
    """
    lang = _detect_language(user_message)
    lang_key = f"message_{lang}"

    # Update history
    updated_history = list(chat_history)
    updated_history.append({"role": "user", "content": user_message})

    updated_data = dict(triage_data)
    should_trigger = False
    next_step = triage_step

    if triage_step == 0:
        # Process greeting response
        updated_data["language"] = lang
        # Check for option selection
        if any(kw in user_message.lower() for kw in ["1", "suspicious", "संदिग्ध"]):
            updated_data["urgency"] = "medium"
        elif any(kw in user_message.lower() for kw in ["2", "digital arrest", "डिजिटल अरेस्ट", "arrest", "धमकी"]):
            updated_data["urgency"] = "high"
        elif any(kw in user_message.lower() for kw in ["3", "transfer", "ट्रांसफर", "money", "पैसे"]):
            updated_data["urgency"] = "critical"
        elif any(kw in user_message.lower() for kw in ["4", "check", "चेक"]):
            updated_data["urgency"] = "low"
        else:
            updated_data["initial_description"] = user_message

        next_step = 1
        response = TRIAGE_STEPS[1][lang_key]

    elif triage_step == 1:
        # Process scam type
        type_map = {
            "1": "video_call_impersonation",
            "2": "phone_call_threat",
            "3": "sms_phishing",
            "4": "email_fraud",
            "5": "other",
        }
        detected_type = type_map.get(user_message.strip(), "other")
        for key, val in type_map.items():
            if key in user_message:
                detected_type = val
                break

        updated_data["scam_type"] = detected_type
        next_step = 2
        response = TRIAGE_STEPS[2][lang_key]

    elif triage_step == 2:
        # Collect details
        updated_data["details"] = user_message
        next_step = 3
        response = TRIAGE_STEPS[3][lang_key]
        should_trigger = True  # Trigger the full KAVACH pipeline on the details

    elif triage_step == 3:
        # Show processing state, move to action plan
        next_step = 4
        risk_summary = _generate_risk_summary(updated_data, risk_score)

        freeze_info = ""
        if risk_score >= 0.65:
            freeze_info = (
                "🛡️ **KAVACH has auto-generated a freeze order request** based on your report. "
                "This can be submitted to CFCFRMS for immediate mule account freezing."
                if lang == "en" else
                "🛡️ **KAVACH ने आपकी रिपोर्ट के आधार पर फ्रीज ऑर्डर तैयार कर दिया है।** "
                "इसे CFCFRMS को तुरंत भेजा जा सकता है।"
            )

        response = TRIAGE_STEPS[4][lang_key].format(
            risk_assessment=risk_summary,
            freeze_info=freeze_info,
        )

    else:
        # Step 5+: Free conversation with guard rails
        response = (
            "If you need more help, please call **1930** immediately. "
            "You can also start a new analysis by sending another message."
            if lang == "en" else
            "अधिक मदद के लिए **1930** पर कॉल करें। "
            "नया विश्लेषण शुरू करने के लिए कोई और मैसेज भेजें।"
        )

    updated_history.append({"role": "assistant", "content": response})

    return {
        "response": response,
        "triage_step": next_step,
        "triage_data": updated_data,
        "chat_history": updated_history,
        "should_trigger_pipeline": should_trigger,
    }
