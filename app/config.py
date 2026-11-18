"""Configuración de DocuAsk leída de variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DOCUASK_DATA_DIR", BASE_DIR / "data"))

PROVIDERS = ("offline", "groq", "openai", "ollama")


def _detect_provider() -> str:
    """Si no se fuerza un proveedor, se elige el primero con credenciales."""
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit.lower()
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "offline"


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=_detect_provider)
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    )
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))

    @property
    def default_model(self) -> str:
        if self.model:
            return self.model
        return {
            "groq": "openai/gpt-oss-120b",
            "openai": "gpt-4o-mini",
            "ollama": "llama3.1",
            "offline": "extractivo-local",
        }.get(self.provider, "openai/gpt-oss-120b")


settings = Settings()
