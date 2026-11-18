from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["provider"] == "offline"
    assert res.json()["status"] == "ok"


def test_flujo_completo_subir_preguntar_borrar():
    res = client.post(
        "/api/documents",
        files={"file": ("notas.md", b"El ciclo DAW dura 2000 horas en total.", "text/markdown")},
    )
    assert res.status_code == 201
    document_id = res.json()["id"]

    ask = client.post("/api/ask", json={"question": "¿Cuántas horas dura el ciclo DAW?"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["sources"]
    assert "2000" in body["answer"]

    listing = client.get("/api/documents").json()
    assert any(d["id"] == document_id for d in listing)

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.delete(f"/api/documents/{document_id}").status_code == 404


def test_rechaza_extensiones_no_soportadas():
    res = client.post(
        "/api/documents",
        files={"file": ("imagen.png", b"\x89PNG", "image/png")},
    )
    assert res.status_code == 415


def test_pregunta_demasiado_corta():
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422


def test_documento_de_ejemplo():
    res = client.post("/api/documents/sample")
    assert res.status_code in (201, 409)


def test_stream_sse():
    with client.stream("GET", "/api/ask/stream", params={"question": "¿Qué es el ciclo DAW?"}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        chunks = "".join(res.iter_text())
    assert "event" not in chunks  # usamos el formato `data: {...}`
    assert '"type": "token"' in chunks or '"type":"token"' in chunks
    assert "done" in chunks
