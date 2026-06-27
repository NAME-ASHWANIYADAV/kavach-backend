"""
KAVACH 2.0 — LangGraph Orchestrator
=====================================
The multi-agent pipeline that routes user input through:
    1. Scam Intel Agent (NLP + RAG → risk score + entities)
    2. Fraud Graph Agent (entity lookup → mule network)
    3. Freeze Architect Agent (auto-generate freeze order)

Key Design Decision:
    All agents share a single KavachState. When Agent 1 extracts
    a phone number, it's AUTOMATICALLY available to Agent 2.
    This is emergent coordination, not sequential execution.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime
from typing import Literal

from langgraph.graph import StateGraph, END

from orchestrator.state import KavachState, AgentLogEntry


logger = logging.getLogger("kavach.orchestrator")


def _add_log(state: KavachState, agent: str, action: str, detail: str = "", status: str = "running") -> list:
    """Append a log entry to the agent activity log."""
    logs = list(state.get("agent_logs", []))
    entry = AgentLogEntry(
        timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        agent_name=agent,
        action=action,
        detail=detail,
        status=status,
    )
    logs.append(entry.to_dict())
    return logs


# ─── Node Functions ─────────────────────────────────────────────────

def scam_intel_node(state: KavachState) -> dict:
    """
    Agent 1: Scam Intelligence
    - Analyze input text for scam indicators
    - Generate risk score (0-100)
    - Extract entities (phone, UPI, accounts)
    - Retrieve RAG citations from government advisories
    """
    from agents.scam_intel import analyze_scam_message

    logs = _add_log(state, "SCAM_INTEL", "Analyzing message for scam patterns...", status="running")

    try:
        result = analyze_scam_message(state["user_input"])

        logs = _add_log(
            {**state, "agent_logs": logs},
            "SCAM_INTEL",
            f"Risk Score: {result['risk_score']:.0%}",
            detail=f"Scam Type: {result['scam_type']} | Entities: {len(result['extracted_entities'])} found",
            status="completed",
        )

        return {
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "scam_type": result["scam_type"],
            "scam_analysis": result["scam_analysis"],
            "extracted_entities": result["extracted_entities"],
            "rag_citations": result["rag_citations"],
            "current_agent": "scam_intel",
            "agent_logs": logs,
        }
    except Exception as e:
        logger.error(f"Scam Intel Agent error: {e}")
        logs = _add_log(
            {**state, "agent_logs": logs},
            "SCAM_INTEL",
            f"Error: {str(e)[:100]}",
            status="error",
        )
        return {
            "risk_score": 0.0,
            "risk_level": "ERROR",
            "scam_analysis": f"Analysis failed: {str(e)}",
            "current_agent": "scam_intel",
            "agent_logs": logs,
            "error_message": str(e),
        }


def fraud_graph_node(state: KavachState) -> dict:
    """
    Agent 2: Fraud Network Graph
    - Take extracted entities from Agent 1
    - Look up in pre-computed fraud network
    - If match: return subgraph with mule cluster
    - Compute centrality metrics
    """
    from agents.fraud_graph import analyze_fraud_network

    entities = state.get("extracted_entities", [])
    logs = _add_log(
        state, "FRAUD_GRAPH",
        f"Querying fraud network for {len(entities)} entities...",
        status="running",
    )

    try:
        result = analyze_fraud_network(entities)

        detail = f"Nodes: {len(result['graph_nodes'])} | Communities: {len(result['graph_communities'])}"
        if result.get("high_centrality_nodes"):
            detail += f" | ⚠ High-centrality nodes: {len(result['high_centrality_nodes'])}"

        logs = _add_log(
            {**state, "agent_logs": logs},
            "FRAUD_GRAPH",
            "Mule Ring identified" if result["graph_match_found"] else "Network analysis complete",
            detail=detail,
            status="completed",
        )

        return {
            "graph_nodes": result["graph_nodes"],
            "graph_edges": result["graph_edges"],
            "graph_communities": result["graph_communities"],
            "mule_ring_summary": result["mule_ring_summary"],
            "high_centrality_nodes": result["high_centrality_nodes"],
            "graph_match_found": result["graph_match_found"],
            "current_agent": "fraud_graph",
            "agent_logs": logs,
        }
    except Exception as e:
        logger.error(f"Fraud Graph Agent error: {e}")
        logs = _add_log(
            {**state, "agent_logs": logs},
            "FRAUD_GRAPH",
            f"Error: {str(e)[:100]}",
            status="error",
        )
        return {
            "graph_nodes": [],
            "graph_edges": [],
            "graph_match_found": False,
            "current_agent": "fraud_graph",
            "agent_logs": logs,
        }


def freeze_architect_node(state: KavachState) -> dict:
    """
    Agent 3: Freeze Architect (THE DIFFERENTIATOR)
    - Take risk score, entities, graph data, and citations
    - Auto-generate BNSS/PMLA-compliant freeze order
    - Create structured document ready for CFCFRMS submission
    """
    from agents.freeze_architect import generate_freeze_order

    risk_score = state.get("risk_score", 0.0)
    logs = _add_log(
        state, "FREEZE_ARCHITECT",
        "Drafting BNSS Section 94 compliant freeze request...",
        status="running",
    )

    try:
        result = generate_freeze_order(state)

        amount_at_risk = "Unknown"
        for entity in state.get("extracted_entities", []):
            if isinstance(entity, dict) and entity.get("entity_type") == "amount":
                amount_at_risk = entity["value"]
                break

        logs = _add_log(
            {**state, "agent_logs": logs},
            "FREEZE_ARCHITECT",
            f"₹{amount_at_risk} AT RISK → FREEZE ORDER READY",
            detail=f"Order ID: {result['freeze_order']['order_id']} | Status: {result['freeze_order']['status']}",
            status="completed",
        )

        return {
            "freeze_order": result["freeze_order"],
            "freeze_order_pdf_path": result.get("pdf_path"),
            "current_agent": "freeze_architect",
            "agent_logs": logs,
        }
    except Exception as e:
        logger.error(f"Freeze Architect error: {e}")
        logs = _add_log(
            {**state, "agent_logs": logs},
            "FREEZE_ARCHITECT",
            f"Error: {str(e)[:100]}",
            status="error",
        )
        return {
            "freeze_order": None,
            "current_agent": "freeze_architect",
            "agent_logs": logs,
        }


def finalize_node(state: KavachState) -> dict:
    """Final node: mark pipeline as completed and compute timing."""
    logs = _add_log(
        state, "SYSTEM",
        "Pipeline complete",
        detail=f"Risk: {state.get('risk_level', 'N/A')} | Freeze Order: {'READY' if state.get('freeze_order') else 'N/A'}",
        status="completed",
    )

    return {
        "pipeline_status": "completed",
        "current_agent": "finalized",
        "agent_logs": logs,
    }


# ─── Routing Logic ──────────────────────────────────────────────────

def should_run_graph(state: KavachState) -> Literal["fraud_graph", "finalize"]:
    """Route to fraud graph only if entities were extracted and risk is non-trivial."""
    entities = state.get("extracted_entities", [])
    risk_score = state.get("risk_score", 0.0)

    if entities and risk_score > 0.3:
        return "fraud_graph"
    return "finalize"


def should_run_freeze(state: KavachState) -> Literal["freeze_architect", "finalize"]:
    """Route to freeze architect only if risk is HIGH/CRITICAL and graph data exists."""
    risk_score = state.get("risk_score", 0.0)

    if risk_score >= 0.65:
        return "freeze_architect"
    return "finalize"


# ─── Build the Graph ────────────────────────────────────────────────

def build_kavach_graph() -> StateGraph:
    """
    Construct the KAVACH multi-agent pipeline.

    Flow:
        Input → Scam Intel → (if entities found) → Fraud Graph → (if high risk) → Freeze Architect → Finalize
                           → (if no entities)  → Finalize
    """
    workflow = StateGraph(KavachState)

    # Add nodes
    workflow.add_node("scam_intel", scam_intel_node)
    workflow.add_node("fraud_graph", fraud_graph_node)
    workflow.add_node("freeze_architect", freeze_architect_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("scam_intel")

    # Conditional edges
    workflow.add_conditional_edges(
        "scam_intel",
        should_run_graph,
        {
            "fraud_graph": "fraud_graph",
            "finalize": "finalize",
        }
    )

    workflow.add_conditional_edges(
        "fraud_graph",
        should_run_freeze,
        {
            "freeze_architect": "freeze_architect",
            "finalize": "finalize",
        }
    )

    # Terminal edges
    workflow.add_edge("freeze_architect", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Singleton compiled graph
kavach_pipeline = build_kavach_graph()
