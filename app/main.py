"""API HTTP de DocuAsk (FastAPI)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, settings
from .graph import RagPipeline
from .ingest import SUPPORTED, chunk_pages, extract
from .models import AskRequest, AskResponse, DocumentInfo, HealthResponse
from .providers import build_provider
from .store import DocumentStore

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_FILE = BASE_DIR / "sample" / "apuntes-daw.md"

app = FastAPI(
    title="DocuAsk API",
    description="Pregunta en lenguaje natural a tus apuntes. RAG con LangGraph.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DocumentStore(persist=os.getenv("DOCUASK_PERSIST", "1") != "0")
provider = build_provider()
pipeline = RagPipeline(store, provider)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=provider.name,
        model=provider.model,
        documents=len(store.documents),
        chunks=len(store.chunks),
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return store.documents


@app.post("/api/documents", response_model=DocumentInfo, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    filename = file.filename or "documento"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(415, f"Formato no soportado. Usa: {', '.join(sorted(SUPPORTED))}")

    content = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(content) > limit:
        raise HTTPException(413, f"El fichero supera el límite de {settings.max_upload_mb} MB")

    try:
        pages = extract(filename, content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # PDF corrupto, cifrado, etc.
        raise HTTPException(400, f"No se pudo leer el documento: {exc}") from exc

    parts = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not parts:
        raise HTTPException(422, "El documento no contiene texto extraíble")

    return store.add(filename, parts)


@app.post("/api/documents/sample", response_model=DocumentInfo, status_code=201)
def load_sample() -> DocumentInfo:
    if not SAMPLE_FILE.exists():
        raise HTTPException(404, "No hay documento de ejemplo disponible")
    if any(d.filename == SAMPLE_FILE.name for d in store.documents):
        raise HTTPException(409, "El documento de ejemplo ya está cargado")

    content = SAMPLE_FILE.read_bytes()
    parts = chunk_pages(extract(SAMPLE_FILE.name, content), settings.chunk_size, settings.chunk_overlap)
    return store.add(SAMPLE_FILE.name, parts)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    if not store.delete(document_id):
        raise HTTPException(404, "Documento no encontrado")


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    return pipeline.ask(payload.question, payload.top_k, payload.document_ids)


@app.get("/api/ask/stream")
def ask_stream(
    question: str = Query(min_length=3, max_length=1000),
    top_k: int = Query(default=5, ge=1, le=10),
) -> StreamingResponse:
    """Server-Sent Events: el frontend recibe el texto según se genera."""

    def event_stream():
        try:
            for kind, value in pipeline.stream(question, top_k):
                if kind == "sources":
                    payload = {"type": "sources", "sources": [s.model_dump() for s in value]}
                else:
                    payload = {"type": "token", "value": value}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as exc:  # el cliente debe recibir el error, no un stream cortado
            error = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


DATA_DIR.mkdir(parents=True, exist_ok=True)
