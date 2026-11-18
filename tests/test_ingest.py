from app.ingest import Page, chunk_pages, extract, normalize


def test_normalize_colapsa_espacios():
    assert normalize("Hola    mundo \n\n\n\n segunda") == "Hola mundo\n\nsegunda"


def test_extrae_texto_plano():
    pages = extract("notas.md", b"# Titulo\n\nContenido de prueba")
    assert len(pages) == 1
    assert "Contenido" in pages[0].text


def test_rechaza_formatos_no_soportados():
    try:
        extract("foto.png", b"123")
    except ValueError as exc:
        assert "no soportado" in str(exc).lower()
    else:
        raise AssertionError("debería haber lanzado ValueError")


def test_los_chunks_respetan_el_tamano():
    text = "\n\n".join(f"Parrafo numero {i} con contenido suficiente para el test." for i in range(40))
    chunks = chunk_pages([Page(number=1, text=text)], chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for _, chunk in chunks)
    assert all(number == 1 for number, _ in chunks)


def test_conserva_el_numero_de_pagina():
    pages = [Page(number=3, text="Texto de la pagina tres con contenido.")]
    assert chunk_pages(pages)[0][0] == 3


def test_parametros_invalidos():
    pages = [Page(number=None, text="algo")]
    for size, overlap in ((50, 0), (300, 400)):
        try:
            chunk_pages(pages, chunk_size=size, overlap=overlap)
        except ValueError:
            continue
        raise AssertionError(f"debería fallar con size={size}, overlap={overlap}")
