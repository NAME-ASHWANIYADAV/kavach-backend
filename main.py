"""
KAVACH 2.0 — Main Application Entry Point
===========================================
Golden Hour Intelligence Engine for Digital Public Safety

Architecture:
    FastAPI → LangGraph (Multi-Agent Orchestration)
        → Agent 1: Scam Intel (NLP + RAG)
        → Agent 2: Fraud Graph (Mule Network)
        → Agent 3: Freeze Architect (BNSS/PMLA Compliance)
        → Agent 4: Citizen Triage Chatbot

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from api.routes import router


# ─── Logging Setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kavach")


# ─── Startup / Shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("=" * 60)
    logger.info("🛡️  KAVACH 2.0 — Golden Hour Intelligence Engine")
    logger.info(f"   Version: {settings.APP_VERSION}")
    logger.info(f"   Debug: {settings.DEBUG}")
    logger.info(f"   Gemini Model: {settings.GEMINI_MODEL}")
    logger.info("=" * 60)

    # Pre-warm the RAG index on startup
    try:
        from rag.retriever import retrieve_advisories
        logger.info("Pre-warming RAG index...")
        retrieve_advisories("test query", top_k=1)
        logger.info("RAG index ready ✅")
    except Exception as e:
        logger.warning(f"RAG pre-warm failed (will lazy-load): {e}")

    # Pre-load fraud graph
    try:
        from agents.fraud_graph import _load_graph
        logger.info("Pre-loading fraud graph...")
        _load_graph()
        logger.info("Fraud graph ready ✅")
    except Exception as e:
        logger.warning(f"Graph pre-load failed (will lazy-load): {e}")

    # Ensure output directories exist
    Path(settings.base_dir / settings.FREEZE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("🚀 KAVACH 2.0 is ready to protect!")
    logger.info(f"   API: http://localhost:{settings.PORT}/docs")

    yield

    logger.info("KAVACH 2.0 shutting down...")


# ─── FastAPI Application ────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "KAVACH 2.0 is a multi-agent intelligence platform that detects digital arrest scams, "
        "maps fraud networks, and auto-generates BNSS-compliant freeze orders within the "
        "Golden Hour window. Built for the ET AI Hackathon 2.0."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── CORS Middleware ─────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Include Routes ─────────────────────────────────────────────────

app.include_router(router)


# ─── Root Endpoint ──────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "analyze": "POST /api/analyze",
            "chat": "POST /api/chat",
            "stream": "GET /api/stream/{session_id}",
            "freeze_submit": "POST /api/freeze/submit",
            "freeze_download": "GET /api/freeze/{session_id}/download",
            "graph_demo": "GET /api/graph/demo",
            "demo_scenarios": "GET /api/demo/scenarios",
            "health": "GET /api/health",
        },
    }


# ─── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
