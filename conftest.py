"""Configuración de pytest: entorno aislado antes de importar la app."""

import os
import tempfile

os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("DOCUASK_PERSIST", "0")
os.environ.setdefault("DOCUASK_DATA_DIR", tempfile.mkdtemp(prefix="docuask-test-"))
