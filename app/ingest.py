"""Extracción de texto y troceado (chunking) de documentos."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Page:
    number: int | None
    text: str


def extract(filename: str, content: bytes) -> list[Page]:
    """Devuelve el texto del documento página a página (1 sola página en txt/md)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Formato no soportado: {suffix}. Usa {', '.join(sorted(SUPPORTED))}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append(Page(number=i, text=page.extract_text() or ""))
        return pages

    text = content.decode("utf-8", errors="replace")
    return [Page(number=None, text=text)]


def normalize(text: str) -> str:
    """Limpia espacios y saltos de línea repetidos sin perder los párrafos."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[tuple[int | None, str]]:
    """Trocea respetando párrafos y frases, con solapamiento para no cortar ideas.

    Devuelve pares (número_de_página, texto_del_chunk).
    """
    if chunk_size < 100:
        raise ValueError("chunk_size demasiado pequeño (mínimo 100)")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe estar entre 0 y chunk_size")

    chunks: list[tuple[int | None, str]] = []

    for page in pages:
        text = normalize(page.text)
        if not text:
            continue

        buffer = ""
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Un párrafo que ya cabe en el buffer se acumula.
            if len(buffer) + len(paragraph) + 2 <= chunk_size:
                buffer = f"{buffer}\n\n{paragraph}".strip()
                continue

            if buffer:
                chunks.append((page.number, buffer))
                buffer = _tail(buffer, overlap)

            # Párrafos más largos que el tamaño de chunk se parten por frases.
            if len(paragraph) > chunk_size:
                for piece in _split_long(paragraph, chunk_size, overlap):
                    chunks.append((page.number, piece))
                buffer = ""
            else:
                buffer = paragraph

        if buffer:
            chunks.append((page.number, buffer))

    return [(page, text) for page, text in chunks if len(text.strip()) >= 20]


def _tail(text: str, overlap: int) -> str:
    """Última porción del chunk anterior, usada como solapamiento."""
    if overlap <= 0:
        return ""
    return text[-overlap:].lstrip()


def _split_long(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(paragraph):
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            current = (current[-overlap:] + " " + sentence).strip() if overlap else sentence
    if current:
        pieces.append(current)
    return pieces
