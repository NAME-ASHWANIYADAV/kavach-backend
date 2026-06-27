<p align="center">
  <h1 align="center">🛡️ KAVACH 2.0 — Backend</h1>
  <p align="center">
    <strong>Golden Hour Intelligence Engine for Digital Public Safety</strong>
  </p>
  <p align="center">
    Multi-Agent AI Pipeline · FastAPI · LangGraph · Gemini 2.5 Flash · FAISS RAG · NetworkX
  </p>
</p>

---

## What Is KAVACH?

KAVACH (कवच = "Shield") is a multi-agent AI platform that combats India's ₹11,333 crore digital arrest scam epidemic. It detects scam messages, maps fraud networks, and auto-generates legally compliant freeze orders — all within the **Golden Hour** (first 45 minutes after fraud when money can still be recovered).

**This repo contains the backend API and AI pipeline.**

> 🔗 **Frontend Repository**: [kavach](https://github.com/NAME-ASHWANIYADAV/kavach)

---

## Architecture

```
FastAPI Server (Port 8000)
│
├── POST /api/analyze      → Full LangGraph Pipeline
│   ├── Agent 1: SCAM INTEL     (NLP + Gemini 2.5 Flash + FAISS RAG)
│   ├── Agent 2: FRAUD GRAPH    (NetworkX + Louvain Community Detection)
│   └── Agent 3: FREEZE ARCHITECT (BNSS/PMLA Compliant DOCX Generation)
│
├── POST /api/chat         → Bilingual Triage Chatbot (5-step)
├── GET  /api/stream/{id}  → SSE Real-time Agent Activity Stream
├── POST /api/freeze/submit → CFCFRMS Submission (Mock)
├── GET  /api/freeze/{id}/download → Download Freeze Order DOCX
├── GET  /api/graph/demo   → Pre-computed Fraud Network Graph
├── GET  /api/demo/scenarios → Demo Preset Scam Scenarios
└── GET  /api/health       → Health Check
```

### Multi-Agent Pipeline Flow

```
Input Message → SCAM INTEL → FRAUD GRAPH → FREEZE ARCHITECT → Response
                   │              │               │
                   ▼              ▼               ▼
              Risk Score     Mule Network    Freeze Order
              Entities       Communities     DOCX Document
              RAG Citations  Hub Nodes       CFCFRMS Reference
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **AI Orchestration** | LangGraph StateGraph |
| **LLM** | Google Gemini 2.5 Flash |
| **RAG** | FAISS + Sentence-Transformers (all-MiniLM-L6-v2) |
| **Graph Analysis** | NetworkX + python-louvain |
| **Document Gen** | python-docx |
| **Streaming** | Server-Sent Events (SSE) |
| **Config** | Pydantic Settings + python-dotenv |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/NAME-ASHWANIYADAV/kavach-backend.git
cd kavach-backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
DEBUG=true
```

> 🔑 Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey)

### 4. Run

```bash
python main.py
```

Server starts at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Project Structure

```
kavach-backend/
├── main.py                 # FastAPI application entry point
├── config.py               # Pydantic settings (env vars)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
│
├── api/
│   └── routes.py           # All REST + SSE endpoints
│
├── orchestrator/
│   ├── state.py            # KavachState TypedDict (shared agent state)
│   └── graph.py            # LangGraph StateGraph pipeline definition
│
├── agents/
│   ├── scam_intel.py       # Agent 1: NLP + RAG scam analysis
│   ├── fraud_graph.py      # Agent 2: NetworkX mule network detection
│   ├── freeze_architect.py # Agent 3: BNSS/PMLA freeze order generation
│   └── chatbot.py          # Agent 4: Bilingual triage chatbot
│
├── rag/
│   └── retriever.py        # FAISS vector index + advisory retrieval
│
├── data/
│   ├── advisories/         # 10 MHA/RBI/I4C advisory text files
│   ├── graph/              # Pre-built mule network (nodes.json, edges.json)
│   └── faiss_index/        # Generated FAISS index (auto-created)
│
├── templates/
│   └── freeze_order_template.docx  # Legal document template
│
└── output/
    └── freeze_orders/      # Generated DOCX files (gitignored)
```

---

## API Reference

### POST `/api/analyze`

Runs the full 3-agent pipeline on a scam message.

**Request:**
```json
{
  "message": "This is CBI. Your Aadhaar is linked to money laundering...",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "risk_score": 0.76,
  "risk_level": "HIGH",
  "scam_type": "digital_arrest",
  "scam_analysis": "Pattern analysis detected digital_arrest indicators.",
  "extracted_entities": [
    {"entity_type": "phone", "value": "+91-9334567890", "confidence": 0.85},
    {"entity_type": "upi_id", "value": "verify.cbi@ybl", "confidence": 0.85},
    {"entity_type": "amount", "value": "₹2,34,000", "confidence": 0.85}
  ],
  "rag_citations": [
    {"source_document": "I4C Advisory: Common Scam Scripts (2024)", "similarity_score": 0.36}
  ],
  "freeze_order": {
    "order_id": "CFCFRMS-2026-xxxx",
    "status": "DRAFT"
  },
  "agent_logs": [...],
  "processing_time_ms": 6923.0
}
```

### POST `/api/chat`

Bilingual chatbot triage conversation.

### GET `/api/stream/{session_id}`

SSE stream for real-time Agent Activity Visualizer.

### POST `/api/freeze/submit`

Submit freeze order to CFCFRMS gateway (mock).

### GET `/api/freeze/{session_id}/download`

Download freeze order as DOCX.

---

## The Three Agents

### 🔍 Agent 1: Scam Intel (`agents/scam_intel.py`)
- **Hybrid entity extraction**: Regex (Aadhaar, PAN, UPI, IFSC, phone) + Gemini semantic extraction
- **Weighted risk scoring**: 40% pattern match + 30% LLM analysis + 30% RAG similarity
- **RAG retrieval**: FAISS index of 10 government advisories
- **Scam classification**: Digital Arrest, UPI Fraud, Customs Parcel, Impersonation

### 🕸️ Agent 2: Fraud Graph (`agents/fraud_graph.py`)
- **NetworkX graph** of known mule account networks
- **Louvain community detection** for mule ring identification
- **Betweenness centrality** for hub node detection
- Visual topology data for frontend graph rendering

### 📜 Agent 3: Freeze Architect (`agents/freeze_architect.py`)
- **BNSS Section 94** + **PMLA Section 17A** compliant orders
- Auto-populated with victim details, suspect accounts, evidence citations
- **DOCX generation** using python-docx
- Ready for CFCFRMS submission

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | **Required.** Gemini API key |
| `DEBUG` | `true` | Enable debug logging |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary LLM model |
| `GEMINI_FLASH_MODEL` | `gemini-2.5-flash` | Chatbot LLM model |
| `GEMINI_TEMPERATURE` | `0.3` | LLM temperature |
| `RAG_TOP_K` | `3` | Number of RAG results |

---

## Built For

**ET AI Hackathon 2.0** — Theme: Digital Public Safety

> *KAVACH is not a chatbot. It's an intelligence platform that turns scattered fraud evidence into actionable legal directives within the Golden Hour.*

---

## License

Apache 2.0
