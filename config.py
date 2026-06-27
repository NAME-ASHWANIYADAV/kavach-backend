"""
KAVACH 2.0 — Configuration Module
=================================
Centralized configuration using Pydantic Settings.
All secrets loaded from environment variables (.env file).
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application-wide settings. Loaded from .env file automatically."""

    # --- Application ---
    APP_NAME: str = "KAVACH 2.0 — Golden Hour Intelligence Engine"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000", "*"]

    # --- Gemini API ---
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.3
    GEMINI_MAX_TOKENS: int = 4096

    # --- RAG Configuration ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    ADVISORIES_DIR: str = "data/advisories"
    RAG_TOP_K: int = 3

    # --- Graph Configuration ---
    GRAPH_NODES_PATH: str = "data/graph/nodes.json"
    GRAPH_EDGES_PATH: str = "data/graph/edges.json"
    GRAPH_COMMUNITIES_PATH: str = "data/graph/communities.json"

    # --- Freeze Order ---
    FREEZE_TEMPLATE_PATH: str = "templates/freeze_order_template.docx"
    FREEZE_OUTPUT_DIR: str = "output/freeze_orders"

    # --- Agent Thresholds ---
    HIGH_RISK_THRESHOLD: float = 0.75
    MEDIUM_RISK_THRESHOLD: float = 0.40
    CENTRALITY_ALERT_THRESHOLD: float = 0.15

    # --- Paths (computed) ---
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def templates_dir(self) -> Path:
        return self.base_dir / "templates"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
settings = Settings()
