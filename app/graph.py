"""Orquestación del RAG con LangGraph.

Grafo:

    START → retrieve ─┬─(hay fragmentos)→ generate → verify → END
                      └─(no hay)─────────→ no_answer ─────────→ END

Cada nodo es una función pura sobre el estado, lo que hace el flujo fácil de leer,
de testear y de ampliar (por ejemplo añadiendo un nodo de reescritura de la pregunta).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .models import AskResponse, SourceRef
from .providers import LLMProvider, build_prompt
from .retrieval import Scored, to_sources
from .store import DocumentStore

NO_ANSWER = "No encuentro esa información en los documentos."


class RagState(TypedDict, total=False):
    question: str
    top_k: int
    document_ids: list[str] | None
    candidates: list[Scored]
    context: str
    answer: str
    grounded: bool


def format_context(candidates: list[Scored]) -> str:
    blocks = []
    for i, item in enumerate(candidates, start=1):
        location = f", pág. {item.chunk.page}" if item.chunk.page else ""
        blocks.append(f"[{i}] ({item.chunk.filename}{location})\n{item.chunk.text}")
    return "\n\n".join(blocks)


class RagPipeline:
    """Fachada que usan la API y los tests."""

    def __init__(self, store: DocumentStore, provider: LLMProvider) -> None:
        self.store = store
        self.provider = provider
        self.graph = self._build_graph()

    # ---------- grafo ----------

    def _build_graph(self):
        builder = StateGraph(RagState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("verify", self._verify)
        builder.add_node("no_answer", self._no_answer)

        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            lambda state: "generate" if state.get("candidates") else "no_answer",
            {"generate": "generate", "no_answer": "no_answer"},
        )
        builder.add_edge("generate", "verify")
        builder.add_edge("verify", END)
        builder.add_edge("no_answer", END)
        return builder.compile()

    def _retrieve(self, state: RagState) -> RagState:
        candidates = self.store.search(
            state["question"],
            top_k=state.get("top_k", 5),
            document_ids=state.get("document_ids"),
        )
        return {"candidates": candidates, "context": format_context(candidates)}

    def _generate(self, state: RagState) -> RagState:
        messages = build_prompt(state["question"], state.get("context", ""))
        return {"answer": self.provider.complete(messages)}

    def _verify(self, state: RagState) -> RagState:
        answer = (state.get("answer") or "").strip()
        if not answer:
            return {"answer": NO_ANSWER, "grounded": False}
        return {"answer": answer, "grounded": bool(re.search(r"\[\d+\]", answer))}

    def _no_answer(self, _state: RagState) -> RagState:
        return {"answer": NO_ANSWER, "grounded": False, "candidates": []}

    # ---------- API pública ----------

    def ask(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> AskResponse:
        started = time.perf_counter()
        state: RagState = self.graph.invoke(
            {"question": question, "top_k": top_k, "document_ids": document_ids}
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        sources = to_sources(state.get("candidates", []))
        return AskResponse(
            answer=state.get("answer", NO_ANSWER),
            sources=sources,
            provider=self.provider.name,
            model=self.provider.model,
            elapsed_ms=elapsed,
            grounded=state.get("grounded", False),
        )

    def stream(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> Iterator[tuple[str, str | list[SourceRef]]]:
        """Genera tokens y, al final, la lista de fuentes.

        Se reutiliza el mismo nodo de recuperación del grafo, de modo que el texto
        que ve el usuario y las citas siempre corresponden al mismo contexto.
        """
        state = self._retrieve({"question": question, "top_k": top_k, "document_ids": document_ids})
        candidates: list[Scored] = state.get("candidates", [])
        sources = to_sources(candidates)

        if not candidates:
            yield ("token", NO_ANSWER)
            yield ("sources", sources)
            return

        messages = build_prompt(question, state["context"])
        for token in self.provider.stream(messages):
            yield ("token", token)
        yield ("sources", sources)
