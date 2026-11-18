from app.models import Chunk
from app.retrieval import BM25Index, reciprocal_rank_fusion, to_sources, tokenize


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"c{i}", document_id="d1", filename="doc.md", index=i, text=text)


def test_tokenize_quita_acentos_y_palabras_vacias():
    assert tokenize("Programación de Aplicaciones Web") == ["programacion", "aplicaciones", "web"]


def test_bm25_ordena_por_relevancia():
    chunks = [
        make_chunk(0, "El ciclo DAW tiene una duración de 2000 horas."),
        make_chunk(1, "React es una librería de JavaScript para interfaces."),
        make_chunk(2, "Docker permite empaquetar aplicaciones en contenedores."),
    ]
    results = BM25Index(chunks).search("¿Cuántas horas dura el ciclo DAW?", top_k=2)
    assert results
    assert results[0].chunk.id == "c0"
    assert 0 < results[0].score


def test_bm25_ignora_consultas_sin_coincidencias():
    chunks = [make_chunk(0, "Contenido sobre bases de datos relacionales.")]
    assert BM25Index(chunks).search("astronomía cuántica") == []


def test_rrf_fusiona_listas():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert list(fused)[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_to_sources_recorta_extractos_largos():
    chunk = make_chunk(0, "x" * 500)
    source = to_sources([type("S", (), {"chunk": chunk, "score": 1.0})()])[0]
    assert source.excerpt.endswith("…")
    assert len(source.excerpt) == 321
