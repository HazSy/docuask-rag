"""Almacén en memoria de documentos y chunks, con persistencia en disco."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from .config import DATA_DIR
from .models import Chunk, DocumentInfo
from .retrieval import BM25Index

INDEX_FILE = DATA_DIR / "index.json"


class DocumentStore:
    """Índice en memoria. Suficiente para un corpus pequeño (< 10 000 chunks).

    Si el corpus crece, esta clase es el único punto a sustituir por pgvector
    o Chroma: el resto de la aplicación solo usa `add`, `search` y `chunks`.
    """

    def __init__(self, persist: bool = True) -> None:
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._documents: dict[str, DocumentInfo] = {}
        self._index = BM25Index([])
        self._persist = persist
        if persist:
            self._load()

    # ---------- lectura ----------

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def documents(self) -> list[DocumentInfo]:
        return list(self._documents.values())

    def search(self, question: str, top_k: int = 5, document_ids: list[str] | None = None):
        with self._lock:
            if document_ids:
                subset = [c for c in self._chunks if c.document_id in document_ids]
                index = BM25Index(subset)
            else:
                index = self._index
        return index.search(question, top_k)

    # ---------- escritura ----------

    def add(self, filename: str, parts: list[tuple[int | None, str]]) -> DocumentInfo:
        document_id = uuid.uuid4().hex[:12]
        with self._lock:
            for i, (page, text) in enumerate(parts):
                self._chunks.append(
                    Chunk(
                        id=f"{document_id}-{i}",
                        document_id=document_id,
                        filename=filename,
                        page=page,
                        index=i,
                        text=text,
                    )
                )
            info = DocumentInfo(
                id=document_id,
                filename=filename,
                chunks=len(parts),
                characters=sum(len(t) for _, t in parts),
            )
            self._documents[document_id] = info
            self._reindex()
        self._save()
        return info

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False
            self._chunks = [c for c in self._chunks if c.document_id != document_id]
            del self._documents[document_id]
            self._reindex()
        self._save()
        return True

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._documents = {}
            self._reindex()
        self._save()

    # ---------- internos ----------

    def _reindex(self) -> None:
        self._index = BM25Index(self._chunks)

    def _save(self) -> None:
        if not self._persist:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [d.model_dump(mode="json") for d in self._documents.values()],
            "chunks": [c.model_dump(mode="json") for c in self._chunks],
        }
        tmp = Path(str(INDEX_FILE) + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(INDEX_FILE)

    def _load(self) -> None:
        if not INDEX_FILE.exists():
            return
        try:
            payload = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**c) for c in payload.get("chunks", [])]
            self._documents = {
                d["id"]: DocumentInfo(**d) for d in payload.get("documents", [])
            }
            self._reindex()
        except (json.JSONDecodeError, TypeError, KeyError):
            # Índice corrupto: se empieza de cero en lugar de romper el arranque.
            self._chunks = []
            self._documents = {}
            self._reindex()
