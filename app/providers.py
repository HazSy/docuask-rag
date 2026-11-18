"""Proveedores de LLM intercambiables: Groq, OpenAI, Ollama y un modo local.

Todos implementan la misma interfaz (`complete` / `stream`), así que la aplicación
no sabe contra qué API está hablando. Eso permite testear el pipeline sin red ni
claves y cambiar de proveedor con una variable de entorno.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx

from .config import Settings, settings as default_settings

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre los documentos del usuario.

Reglas:
1. Responde SOLO con información del CONTEXTO. Si no está, di: "No encuentro esa información en los documentos".
2. Cita siempre las fuentes con el formato [1], [2] usando el número de cada fragmento.
3. Responde en español, de forma clara y breve (máximo 6 frases).
4. No inventes datos, cifras ni nombres.
"""


def build_prompt(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}",
        },
    ]


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Por defecto, una única respuesta. Los proveedores reales lo sobreescriben."""
        yield self.complete(messages)


class OfflineProvider(LLMProvider):
    """Modo demo sin API: respuesta extractiva a partir del contexto.

    No es un LLM, pero permite probar la app (y los tests) sin claves ni internet.
    """

    name = "offline"

    def __init__(self, model: str = "extractivo-local") -> None:
        self.model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        context, _, question = user.partition("PREGUNTA:")
        context = context.removeprefix("CONTEXTO:").strip() or question
        q_terms = {w for w in re.findall(r"[a-záéíóúñ]{4,}", question.lower())}

        ranked: list[tuple[int, int, str]] = []
        for block in re.split(r"\n\n(?=\[\d+\])", context):
            match = re.match(r"\[(\d+)\]", block.strip())
            if not match:
                continue
            number = int(match.group(1))
            for sentence in re.split(r"(?<=[.!?])\s+", block):
                score = len(q_terms & set(re.findall(r"[a-záéíóúñ]{4,}", sentence.lower())))
                if score:
                    ranked.append((score, number, " ".join(sentence.split())))

        if not ranked:
            return "No encuentro esa información en los documentos."

        ranked.sort(key=lambda item: item[0], reverse=True)
        picked = ranked[:3]
        body = " ".join(f"{sentence} [{number}]" for _, number, sentence in picked)
        return f"Según los documentos: {body}"

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        for word in self.complete(messages).split(" "):
            yield word + " "


class _ChatCompletionsProvider(LLMProvider):
    """Base para APIs compatibles con el formato de OpenAI."""

    base_url: str
    api_key: str

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "stream": True,
            },
            timeout=60,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if content := delta.get("content"):
                    yield content


class GroqProvider(_ChatCompletionsProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class OpenAIProvider(_ChatCompletionsProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, url: str) -> None:
        self.model = model
        self.url = url.rstrip("/")

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": True},
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if content := payload.get("message", {}).get("content"):
                    yield content


def build_provider(config: Settings | None = None) -> LLMProvider:
    config = config or default_settings
    provider = config.provider

    if provider == "groq":
        return GroqProvider(config.default_model, config.api_key)
    if provider == "openai":
        return OpenAIProvider(config.default_model, config.api_key)
    if provider == "ollama":
        return OllamaProvider(config.default_model, config.ollama_url)
    return OfflineProvider(config.default_model)
