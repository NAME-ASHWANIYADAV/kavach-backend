"""
KAVACH 2.0 — Shared Agent State
================================
The KavachState TypedDict is the single source of truth shared across
all agents in the LangGraph pipeline. Every agent reads from and writes
to this state, enabling true multi-agent coordination.

Architecture Insight:
    When Agent 1 (Scam Intel) extracts a phone number, it's automatically
    available to Agent 2 (Fraud Graph). When Agent 2 finds a mule cluster,
    Agent 3 (Freeze Architect) uses it to draft the freeze order.
    This is NOT sequential execution — it's emergent coordination.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime


# ─── Entity Types ───────────────────────────────────────────────────
@dataclass
class ExtractedEntity:
    """A named entity extracted from scam text analysis."""
    entity_type: str          # "phone", "upi_id", "account_number", "name", "amount"
    value: str                # The actual extracted value
    confidence: float = 0.0   # 0.0 to 1.0
    source_text: str = ""     # The original text fragment it was extracted from

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "confidence": self.confidence,
            "source_text": self.source_text,
        }


@dataclass
class RAGCitation:
    """A citation from RAG retrieval over government advisories."""
    source_document: str      # e.g., "MHA Advisory #2024-47"
    relevant_text: str        # The cited paragraph
    similarity_score: float   # Cosine similarity score
    advisory_date: str = ""   # Date of the advisory
    issuing_authority: str = ""  # "MHA", "RBI", "I4C", etc.

    def to_dict(self) -> dict:
        return {
            "source_document": self.source_document,
            "relevant_text": self.relevant_text,
            "similarity_score": self.similarity_score,
            "advisory_date": self.advisory_date,
            "issuing_authority": self.issuing_authority,
        }


@dataclass
class GraphNode:
    """A node in the fraud network graph."""
    node_id: str
    label: str                # Display label (masked phone/UPI)
    node_type: str            # "victim", "mule_l1", "mule_l2", "collector", "suspect"
    community_id: int = -1    # Louvain community assignment
    betweenness: float = 0.0  # Betweenness centrality score
    is_flagged: bool = False  # Whether this node is flagged as high-risk
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "community_id": self.community_id,
            "betweenness": self.betweenness,
            "is_flagged": self.is_flagged,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """An edge in the fraud network graph."""
    source: str               # Source node ID
    target: str               # Target node ID
    edge_type: str            # "transfer", "call", "linked_account"
    weight: float = 1.0       # Edge weight (transaction frequency/amount)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class FreezeOrder:
    """A BNSS/PMLA-compliant freeze order document."""
    order_id: str
    generated_at: str
    victim_name: str = "Anonymous Complainant"
    victim_contact: str = ""
    suspect_entities: list = field(default_factory=list)  # List of suspect UPI/phone/account
    fraud_amount: str = "Unknown"
    fraud_type: str = "Digital Arrest Scam"
    risk_score: float = 0.0
    evidence_citations: list = field(default_factory=list)  # RAGCitation list
    mule_network_summary: str = ""
    legal_sections: list = field(default_factory=lambda: [
        "BNSS Section 94 (Search & Seizure of Digital Evidence)",
        "PMLA Section 17A (Attachment of Property in Money Laundering)",
        "IT Act Section 69 (Interception/Monitoring of Digital Information)",
    ])
    cfcfrms_reference: str = ""
    status: str = "DRAFT"     # DRAFT, SUBMITTED, ACKNOWLEDGED

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "generated_at": self.generated_at,
            "victim_name": self.victim_name,
            "victim_contact": self.victim_contact,
            "suspect_entities": self.suspect_entities,
            "fraud_amount": self.fraud_amount,
            "fraud_type": self.fraud_type,
            "risk_score": self.risk_score,
            "evidence_citations": [
                c.to_dict() if hasattr(c, 'to_dict') else c
                for c in self.evidence_citations
            ],
            "mule_network_summary": self.mule_network_summary,
            "legal_sections": self.legal_sections,
            "cfcfrms_reference": self.cfcfrms_reference,
            "status": self.status,
        }


@dataclass
class AgentLogEntry:
    """A single entry in the Agent Activity Visualizer log."""
    timestamp: str
    agent_name: str           # "SCAM_INTEL", "FRAUD_GRAPH", "FREEZE_ARCHITECT"
    action: str               # What the agent is doing
    detail: str = ""          # Additional detail
    status: str = "running"   # "running", "completed", "error"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "action": self.action,
            "detail": self.detail,
            "status": self.status,
        }


# ─── The Core State ─────────────────────────────────────────────────
class KavachState(TypedDict, total=False):
    """
    The unified state shared across all KAVACH agents.
    
    This is the beating heart of the multi-agent system.
    Every agent reads from and enriches this state.
    LangGraph manages state transitions automatically.
    """

    # --- Input ---
    user_input: str                           # Raw user message or scam text
    input_type: str                           # "scam_check", "chat", "graph_query"
    session_id: str                           # Unique session identifier

    # --- Agent 1: Scam Intel Output ---
    risk_score: float                         # 0.0 to 1.0 (displayed as 0-100)
    risk_level: str                           # "CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE"
    scam_type: str                            # "digital_arrest", "upi_fraud", "phishing", etc.
    scam_analysis: str                        # LLM-generated analysis summary
    extracted_entities: list                   # List of ExtractedEntity dicts
    rag_citations: list                       # List of RAGCitation dicts

    # --- Agent 2: Fraud Graph Output ---
    graph_nodes: list                         # List of GraphNode dicts
    graph_edges: list                         # List of GraphEdge dicts
    graph_communities: list                   # List of community IDs detected
    mule_ring_summary: str                    # Summary of detected mule ring
    high_centrality_nodes: list               # Nodes with betweenness > threshold
    graph_match_found: bool                   # Whether entities matched in graph

    # --- Agent 3: Freeze Architect Output ---
    freeze_order: Optional[dict]              # FreezeOrder dict (or None)
    freeze_order_pdf_path: Optional[str]      # Path to generated PDF
    cfcfrms_submitted: bool                   # Whether "submitted" to mock CFCFRMS

    # --- Chatbot State ---
    chat_history: list                        # List of {"role": ..., "content": ...}
    triage_step: int                          # Current step in 5-step triage (0-5)
    triage_data: dict                         # Collected triage information

    # --- Agent Activity Log (for Visualizer) ---
    agent_logs: list                          # List of AgentLogEntry dicts

    # --- Pipeline Control ---
    current_agent: str                        # Which agent is currently executing
    pipeline_status: str                      # "processing", "completed", "error"
    error_message: Optional[str]              # Error details if any
    processing_time_ms: float                 # Total processing time


def create_initial_state(
    user_input: str,
    input_type: str = "scam_check",
    session_id: str = ""
) -> KavachState:
    """Create a fresh KavachState with default values."""
    import uuid

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    return KavachState(
        # Input
        user_input=user_input,
        input_type=input_type,
        session_id=session_id,

        # Agent 1
        risk_score=0.0,
        risk_level="UNKNOWN",
        scam_type="unknown",
        scam_analysis="",
        extracted_entities=[],
        rag_citations=[],

        # Agent 2
        graph_nodes=[],
        graph_edges=[],
        graph_communities=[],
        mule_ring_summary="",
        high_centrality_nodes=[],
        graph_match_found=False,

        # Agent 3
        freeze_order=None,
        freeze_order_pdf_path=None,
        cfcfrms_submitted=False,

        # Chatbot
        chat_history=[],
        triage_step=0,
        triage_data={},

        # Activity Log
        agent_logs=[],

        # Pipeline
        current_agent="initializing",
        pipeline_status="processing",
        error_message=None,
        processing_time_ms=0.0,
    )
