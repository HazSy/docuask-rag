# DocuAsk

> Pregunta en lenguaje natural a tus apuntes. RAG con LangGraph, FastAPI y streaming en tiempo real, con citas a la fuente.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.6-1C3C3C)
![Tests](https://img.shields.io/badge/Tests-24%20passing-6E9F18?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

Subes tus apuntes (PDF, Markdown o TXT) y el asistente responde **solo** con lo que
aparece en ellos, indicando de qué fragmento sale cada afirmación (`[1]`, `[2]`…).
Si la respuesta no está en los documentos, lo dice en lugar de inventarla.

---

## 🎯 Por qué este proyecto

La demanda de perfiles que **integran IA en productos reales** es la que más ha crecido
en 2026. Este proyecto no es un wrapper de ChatGPT: implementa el patrón RAG completo
con recuperación léxica propia, un grafo de estados explícito, verificación de citas y
una abstracción de proveedor que permite funcionar sin API key.

---

## ✨ Funcionalidades

| Función | Detalle |
|---|---|
| 📄 **Ingesta** | PDF (pypdf), Markdown y TXT, con troceado por párrafos y solapamiento |
| 🔎 **Recuperación BM25** | Implementación propia sin dependencias, insensible a acentos y mayúsculas |
| 🧭 **Grafo LangGraph** | `retrieve → (generate → verify) \| no_answer`, con ruta condicional |
| 🧠 **Proveedores** | Groq, OpenAI, Ollama local y un modo *offline* extractivo para demo/tests |
| 📡 **Streaming SSE** | La respuesta llega palabra a palabra, con las fuentes al final |
| ✅ **Anti-alucinación** | El nodo `verify` marca `grounded=false` si no hay citas válidas |
| 🧪 **24 tests** | Ingesta, recuperación, grafo y API, ejecutados en CI sin red ni claves |
| 🖥️ **UI incluida** | Interfaz web servida por la propia API (drag & drop + chat) |

---

## 🏗️ Cómo funciona

```
                    ┌──────────────┐
   pregunta ───────▶│  retrieve    │  BM25 sobre los chunks
                    └──────┬───────┘
                           │ ¿hay fragmentos?
              ┌────────────┴─────────────┐
              │ sí                       │ no
              ▼                          ▼
       ┌──────────────┐          ┌──────────────┐
       │  generate    │  LLM     │  no_answer   │
       └──────┬───────┘          └──────┬───────┘
              ▼                         │
       ┌──────────────┐                 │
       │   verify     │  ¿cita [n]?     │
       └──────┬───────┘                 │
              └───────────┬─────────────┘
                          ▼
                       respuesta + fuentes
```

El contexto que se envía al modelo es una lista numerada de fragmentos; el *system
prompt* obliga a citar esos números. El nodo `verify` comprueba que la respuesta
contenga al menos una cita y, si no, la marca como no fundamentada.

---

## 🚀 Puesta en marcha

```bash
git clone https://github.com/HazSy/docuask-rag.git
cd docuask-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload    # http://localhost:8000
```

Abre <http://localhost:8000>, pulsa **«Cargar apuntes de ejemplo»** y pregunta.

### Conectar un LLM real

Sin configuración funciona en **modo offline** (respuestas extractivas, ideal para
probar la app). Para usar un modelo de verdad:

```bash
# Groq (gratuito)
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_tu_clave

# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Ollama local
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://localhost:11434
export LLM_MODEL=llama3.1
```

| Variable | Por defecto | Descripción |
|---|---|---|
| `LLM_PROVIDER` | autodetectado | `groq`, `openai`, `ollama` u `offline` |
| `LLM_MODEL` | según proveedor | Modelo a usar |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `120` | Troceado de documentos |
| `TOP_K` | `5` | Fragmentos recuperados por consulta |
| `MAX_UPLOAD_MB` | `10` | Tamaño máximo de subida |
| `DOCUASK_PERSIST` | `1` | Guardar el índice en `data/index.json` |

---

## 📡 API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/health` | Estado, proveedor y tamaño del índice |
| `GET` | `/api/documents` | Documentos cargados |
| `POST` | `/api/documents` | Subir documento (`multipart/form-data`) |
| `POST` | `/api/documents/sample` | Cargar los apuntes de ejemplo |
| `DELETE` | `/api/documents/{id}` | Eliminar documento y sus chunks |
| `POST` | `/api/ask` | Preguntar y recibir respuesta + fuentes |
| `GET` | `/api/ask/stream` | Igual, pero con Server-Sent Events |

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Cuántas horas tiene el módulo de FCT?","top_k":4}'
```

```json
{
  "answer": "Según los documentos: ... incluye un módulo de Formación en Centros de Trabajo (FCT) de 400 horas ... [1]",
  "sources": [
    { "chunk_id": "999bcdc9888d-0", "filename": "apuntes-daw.md", "page": null, "score": 6.443, "excerpt": "…" }
  ],
  "provider": "offline",
  "model": "extractivo-local",
  "elapsed_ms": 6,
  "grounded": true
}
```

Documentación interactiva generada por FastAPI en `/docs`.

---

## 🗂️ Estructura

```
docuask-rag/
├── app/
│   ├── config.py       # Settings y detección de proveedor
│   ├── ingest.py       # Extracción (PDF/MD/TXT) y chunking
│   ├── retrieval.py    # BM25, tokenizador y fusión RRF
│   ├── store.py        # Índice en memoria + persistencia JSON
│   ├── providers.py    # Groq / OpenAI / Ollama / offline
│   ├── graph.py        # Grafo LangGraph y fachada RagPipeline
│   └── main.py         # FastAPI + SSE + UI estática
├── static/index.html   # Interfaz de chat
├── sample/             # Documento de ejemplo
└── tests/              # 24 tests (pytest)
```

---

## 🧪 Tests

```bash
pytest
```

Los tests fuerzan `LLM_PROVIDER=offline` y un directorio temporal, así que corren
en CI sin claves, sin red y sin tocar tus datos.

---

## ⚠️ Limitaciones conocidas

- El índice vive en memoria y se serializa a JSON: válido hasta ~10 000 fragmentos.
  Para más volumen, sustituir `DocumentStore` por pgvector o Chroma (es el único
  punto que habría que cambiar).
- La recuperación es léxica (BM25). No captura sinónimos; se compensa con el
  solapamiento de chunks y se puede ampliar con embeddings.
- El modo offline **no es un LLM**: selecciona frases del contexto. Sirve para demo
  y tests, no para producción.

## 🗺️ Roadmap

- [ ] Embeddings + fusión RRF con búsqueda semántica (la función `reciprocal_rank_fusion` ya está lista)
- [ ] Re-ranking con un cross-encoder
- [ ] Historial de conversación con memoria en el grafo
- [ ] Autenticación y colecciones por usuario
- [ ] Despliegue con Docker Compose + volumen persistente

---

## 👤 Autor

**Oscar López Pellicer** — Estudiante de Desarrollo de Aplicaciones Web (Murcia, España)

[![GitHub](https://img.shields.io/badge/GitHub-HazSy-181717?logo=github)](https://github.com/HazSy)

Licencia [MIT](./LICENSE).
