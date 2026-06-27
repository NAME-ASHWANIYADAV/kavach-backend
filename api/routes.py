"""
KAVACH 2.0 — API Routes
=========================
All REST + SSE endpoints for the KAVACH platform.

Endpoints:
    POST /api/analyze       → Full pipeline (scam check → graph → freeze order)
    POST /api/chat          → Chatbot triage conversation
    GET  /api/stream/{id}   → SSE stream for Agent Activity Visualizer
    POST /api/freeze/submit → Mock CFCFRMS submission
    GET  /api/freeze/{id}   → Download freeze order document
    GET  /api/graph/demo    → Get pre-computed demo graph data
    GET  /api/health        → Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from config import settings
from orchestrator.state import KavachState, create_initial_state
from orchestrator.graph import kavach_pipeline
from agents.chatbot import process_chat_message
from agents.freeze_architect import generate_freeze_pdf


logger = logging.getLogger("kavach.api")
router = APIRouter(prefix="/api", tags=["KAVACH API"])

# ─── In-Memory Session Store ────────────────────────────────────────
# In production, use Redis. For hackathon, in-memory dict is fine.
_sessions: dict[str, dict] = {}


# ─── Request/Response Models ────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for /api/analyze"""
    message: str = Field(..., min_length=1, max_length=5000, description="Scam message or text to analyze")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class AnalyzeResponse(BaseModel):
    """Response body for /api/analyze"""
    session_id: str
    risk_score: float
    risk_level: str
    scam_type: str
    scam_analysis: str
    extracted_entities: list
    rag_citations: list
    graph_nodes: list
    graph_edges: list
    graph_communities: list
    mule_ring_summary: str
    high_centrality_nodes: list
    graph_match_found: bool
    freeze_order: Optional[dict]
    agent_logs: list
    processing_time_ms: float


class ChatRequest(BaseModel):
    """Request body for /api/chat"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body for /api/chat"""
    session_id: str
    response: str
    triage_step: int
    chat_history: list
    risk_score: float = 0.0
    should_trigger_pipeline: bool = False
    pipeline_result: Optional[dict] = None


class FreezeSubmitRequest(BaseModel):
    """Request body for /api/freeze/submit"""
    session_id: str
    freeze_order_id: str


class FreezeSubmitResponse(BaseModel):
    """Response body for /api/freeze/submit"""
    success: bool
    cfcfrms_reference: str
    message: str
    submitted_at: str


# ─── Routes ─────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_message(request: AnalyzeRequest):
    """
    Run the full KAVACH pipeline on a message.
    
    Flow: Scam Intel → Fraud Graph → Freeze Architect → Response
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())[:8]

    logger.info(f"[{session_id}] Starting analysis: {request.message[:60]}...")

    try:
        # Create initial state
        initial_state = create_initial_state(
            user_input=request.message,
            input_type="scam_check",
            session_id=session_id,
        )

        # Run the LangGraph pipeline
        final_state = kavach_pipeline.invoke(initial_state)

        processing_time = (time.time() - start_time) * 1000

        # Store session
        _sessions[session_id] = {
            "state": final_state,
            "timestamp": datetime.now().isoformat(),
        }

        response = AnalyzeResponse(
            session_id=session_id,
            risk_score=final_state.get("risk_score", 0.0),
            risk_level=final_state.get("risk_level", "UNKNOWN"),
            scam_type=final_state.get("scam_type", "unknown"),
            scam_analysis=final_state.get("scam_analysis", ""),
            extracted_entities=final_state.get("extracted_entities", []),
            rag_citations=final_state.get("rag_citations", []),
            graph_nodes=final_state.get("graph_nodes", []),
            graph_edges=final_state.get("graph_edges", []),
            graph_communities=final_state.get("graph_communities", []),
            mule_ring_summary=final_state.get("mule_ring_summary", ""),
            high_centrality_nodes=final_state.get("high_centrality_nodes", []),
            graph_match_found=final_state.get("graph_match_found", False),
            freeze_order=final_state.get("freeze_order"),
            agent_logs=final_state.get("agent_logs", []),
            processing_time_ms=round(processing_time, 2),
        )

        logger.info(f"[{session_id}] Analysis complete in {processing_time:.0f}ms — {final_state.get('risk_level')}")
        return response

    except Exception as e:
        logger.error(f"[{session_id}] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the triage chatbot.
    """
    session_id = request.session_id or str(uuid.uuid4())[:8]

    # Get or create session
    session = _sessions.get(session_id, {})
    chat_history = session.get("chat_history", [])
    triage_step = session.get("triage_step", 0)
    triage_data = session.get("triage_data", {})
    risk_score = session.get("state", {}).get("risk_score", 0.0)

    # Process through chatbot
    result = process_chat_message(
        user_message=request.message,
        chat_history=chat_history,
        triage_step=triage_step,
        triage_data=triage_data,
        risk_score=risk_score,
    )

    # Update session
    _sessions[session_id] = {
        **session,
        "chat_history": result["chat_history"],
        "triage_step": result["triage_step"],
        "triage_data": result["triage_data"],
    }

    # If chatbot collected enough data, trigger pipeline
    pipeline_result = None
    if result["should_trigger_pipeline"]:
        try:
            details = result["triage_data"].get("details", request.message)
            initial_state = create_initial_state(
                user_input=details,
                input_type="scam_check",
                session_id=session_id,
            )
            final_state = kavach_pipeline.invoke(initial_state)
            _sessions[session_id]["state"] = final_state
            risk_score = final_state.get("risk_score", 0.0)

            pipeline_result = {
                "risk_score": final_state.get("risk_score", 0.0),
                "risk_level": final_state.get("risk_level", "UNKNOWN"),
                "scam_type": final_state.get("scam_type", "unknown"),
                "extracted_entities": final_state.get("extracted_entities", []),
                "freeze_order": final_state.get("freeze_order"),
            }
        except Exception as e:
            logger.error(f"Pipeline trigger from chat failed: {e}")

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        triage_step=result["triage_step"],
        chat_history=result["chat_history"],
        risk_score=risk_score,
        should_trigger_pipeline=result["should_trigger_pipeline"],
        pipeline_result=pipeline_result,
    )


@router.get("/stream/{session_id}")
async def stream_agent_activity(session_id: str):
    """
    SSE endpoint for real-time Agent Activity Visualizer.
    Streams agent state transitions as Server-Sent Events.
    """
    async def event_generator():
        last_log_count = 0
        heartbeat_count = 0

        while True:
            session = _sessions.get(session_id)

            if session and "state" in session:
                state = session["state"]
                logs = state.get("agent_logs", [])

                # Send new logs
                if len(logs) > last_log_count:
                    for log in logs[last_log_count:]:
                        yield f"data: {json.dumps(log)}\n\n"
                    last_log_count = len(logs)

                # Check if pipeline completed
                if state.get("pipeline_status") == "completed":
                    yield f"data: {json.dumps({'agent_name': 'SYSTEM', 'action': 'Pipeline complete', 'status': 'completed'})}\n\n"
                    break

            # Heartbeat every 3 seconds
            heartbeat_count += 1
            if heartbeat_count % 6 == 0:  # Every 3 seconds (6 * 0.5s)
                yield f": heartbeat\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/freeze/submit", response_model=FreezeSubmitResponse)
async def submit_freeze_order(request: FreezeSubmitRequest):
    """
    Mock CFCFRMS submission endpoint.
    In production, this would connect to the actual CFCFRMS API.
    """
    session = _sessions.get(request.session_id)
    if not session or "state" not in session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = session["state"]
    freeze_order = state.get("freeze_order")
    if not freeze_order:
        raise HTTPException(status_code=400, detail="No freeze order found in session")

    # Simulate CFCFRMS submission
    cfcfrms_ref = f"CFCFRMS-{datetime.now().year}-{uuid.uuid4().hex[:8].upper()}"
    submitted_at = datetime.now().isoformat()

    # Update session
    freeze_order["status"] = "SUBMITTED"
    freeze_order["cfcfrms_reference"] = cfcfrms_ref
    state["cfcfrms_submitted"] = True
    _sessions[request.session_id]["state"] = state

    logger.info(f"Freeze order submitted: {cfcfrms_ref}")

    return FreezeSubmitResponse(
        success=True,
        cfcfrms_reference=cfcfrms_ref,
        message=f"Freeze order successfully submitted to CFCFRMS. Reference: {cfcfrms_ref}. Estimated freeze action within 4 hours.",
        submitted_at=submitted_at,
    )


@router.get("/freeze/{session_id}/download")
async def download_freeze_order(session_id: str):
    """Download the freeze order as a DOCX file."""
    session = _sessions.get(session_id)
    if not session or "state" not in session:
        raise HTTPException(status_code=404, detail="Session not found")

    freeze_order = session["state"].get("freeze_order")
    if not freeze_order:
        raise HTTPException(status_code=400, detail="No freeze order in this session")

    # Generate the document
    output_dir = str(Path(settings.base_dir) / settings.FREEZE_OUTPUT_DIR)
    doc_path = generate_freeze_pdf(freeze_order, output_dir)

    if not doc_path or not Path(doc_path).exists():
        raise HTTPException(status_code=500, detail="Failed to generate document")

    return FileResponse(
        path=doc_path,
        filename=f"KAVACH_Freeze_Order_{freeze_order['order_id']}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get("/graph/demo")
async def get_demo_graph():
    """Return pre-computed demo graph data for visualization."""
    from agents.fraud_graph import analyze_fraud_network

    # Return a demo subgraph with no specific entities
    result = analyze_fraud_network([])

    return {
        "nodes": result["graph_nodes"],
        "edges": result["graph_edges"],
        "communities": result["graph_communities"],
        "summary": result["mule_ring_summary"],
    }


@router.get("/demo/scenarios")
async def get_demo_scenarios():
    """Return pre-loaded demo scenarios for the presentation."""
    return {
        "scenarios": [
            {
                "id": "digital_arrest_cbi",
                "title": "CBI Digital Arrest Scam",
                "description": "Classic digital arrest scam impersonating CBI officer",
                "message": (
                    "This is an urgent message from the Central Bureau of Investigation. "
                    "Your Aadhaar number 8742-XXXX-XXXX is linked to a money laundering case "
                    "involving ₹2,34,000. A FIR has been registered under Section 420 IPC. "
                    "To avoid arrest, transfer the amount to UPI ID: verify.cbi@ybl for "
                    "investigation clearance. Do not inform anyone — this is a confidential "
                    "matter. Contact officer at +91-9334567890 immediately."
                ),
            },
            {
                "id": "upi_kyc_fraud",
                "title": "UPI KYC Update Scam",
                "description": "Phishing message requesting KYC update",
                "message": (
                    "Dear Customer, your SBI bank account will be blocked within 24 hours "
                    "due to incomplete KYC verification. Click here to update your KYC: "
                    "https://sbi-kyc-update.fake.com. Enter your debit card number, CVV, "
                    "and OTP to complete verification. Contact helpdesk at 7061234567 "
                    "for assistance. Amount pending verification: ₹15,000."
                ),
            },
            {
                "id": "customs_parcel_scam",
                "title": "Customs/Parcel Scam",
                "description": "Scam claiming intercepted parcel with illegal contents",
                "message": (
                    "This is Mumbai Customs Department. A parcel addressed to your name "
                    "has been intercepted containing suspicious substances. Your Aadhaar "
                    "and PAN ABCDE1234F are linked to this shipment. Transfer ₹1,50,000 "
                    "to account 9876543210123456 (IFSC: SBIN0001234) as security deposit "
                    "or face immediate arrest under NDPS Act. Call Inspector Sharma at "
                    "+91-9460789012. This is a COURT ORDER — do not tell your family."
                ),
            },
        ]
    }
