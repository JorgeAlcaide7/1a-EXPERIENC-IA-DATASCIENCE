"""Configuración y utilidades compartidas por los tests."""
import ast
from pathlib import Path

import pytest

# Raíz del proyecto (carpeta que contiene app/, data/, models/, ...)
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def root():
    """Ruta a la raíz del proyecto."""
    return ROOT


@pytest.fixture
def app_function():
    """Devuelve un cargador que extrae UNA función de app/app.py.

    No se puede `import app` directamente porque app.py arranca Streamlit al
    importarse (st.set_page_config, carga del modelo, etc.). Por eso extraemos
    solo la función pedida con ast y la compilamos de forma aislada, de modo
    que los tests prueban el código REAL de la app sin levantar la interfaz.
    """
    import numpy as np
    import pandas as pd
    import re

    def _loader(name):
        src = (ROOT / "app" / "app.py").read_text()
        tree = ast.parse(src)
        func = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        ns = {"np": np, "pd": pd, "re": re}
        exec(compile(ast.Module([func], []), "app.py", "exec"), ns)
        return ns[name]

    return _loader
