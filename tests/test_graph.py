from app.graph import NO_ANSWER, RagPipeline
from app.models import Chunk
from app.providers import OfflineProvider
from app.store import DocumentStore

DOC = """El ciclo DAW tiene una duración de 2000 horas repartidas en dos cursos.

El módulo de Formación en Centros de Trabajo tiene 400 horas y se realiza en segundo curso.

La programación se imparte con Java y el desarrollo servidor con PHP.
"""


def build_pipeline() -> RagPipeline:
    store = DocumentStore(persist=False)
    store.add("apuntes.md", [(1, DOC)])
    return RagPipeline(store, OfflineProvider())


def test_responde_con_citas():
    result = build_pipeline().ask("¿Cuántas horas tiene el ciclo DAW?")
    assert result.grounded is True
    assert "[1]" in result.answer or "[2]" in result.answer
    assert result.sources


def test_sin_documentos_devuelve_no_answer():
    pipeline = RagPipeline(DocumentStore(persist=False), OfflineProvider())
    result = pipeline.ask("¿Cuántas horas tiene el ciclo DAW?")
    assert result.answer == NO_ANSWER
    assert result.grounded is False
    assert result.sources == []


def test_pregunta_sin_relacion_no_inventa():
    result = build_pipeline().ask("¿Cómo se cultiva la vid en La Rioja?")
    assert result.answer == NO_ANSWER
    assert result.grounded is False


def test_stream_emite_tokens_y_fuentes():
    events = list(build_pipeline().stream("¿Qué módulo tiene 400 horas?"))
    kinds = [kind for kind, _ in events]
    assert "token" in kinds
    assert kinds[-1] == "sources"
    assert events[-1][1]


def test_filtra_por_documento():
    store = DocumentStore(persist=False)
    a = store.add("a.md", [(None, "El ciclo DAW dura 2000 horas.")])
    store.add("b.md", [(None, "Docker empaqueta aplicaciones en contenedores.")])

    pipeline = RagPipeline(store, OfflineProvider())
    result = pipeline.ask("¿Cuánto dura el ciclo DAW?", document_ids=[a.id])
    assert result.sources
    assert all(s.filename == "a.md" for s in result.sources)


def test_indice_descarta_fragmentos_cortos():
    from app.ingest import Page, chunk_pages

    assert chunk_pages([Page(number=None, text="ok")]) == []


def test_chunk_ids_son_unicos():
    store = DocumentStore(persist=False)
    info = store.add("doc.md", [(1, "Uno dos tres cuatro cinco seis siete."), (2, "Ocho nueve diez once.")])
    ids = [c.id for c in store.chunks]
    assert len(ids) == len(set(ids)) == info.chunks
    assert isinstance(Chunk(**store.chunks[0].model_dump()), Chunk)
