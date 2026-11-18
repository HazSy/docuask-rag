"""Esquemas de entrada y salida de la API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page: int | None = None
    index: int
    text: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int
    characters: int
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SourceRef(BaseModel):
    chunk_id: str
    filename: str
    page: int | None = None
    score: float
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: list[str] | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    provider: str
    model: str
    elapsed_ms: int
    grounded: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    model: str
    documents: int
    chunks: int
