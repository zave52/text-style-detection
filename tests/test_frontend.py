"""Exercise the standalone Streamlit image layout in a fresh process."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("frontend_only", [False, True])
def test_streamlit_starts_without_backend_imports(tmp_path, frontend_only):
    source = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    if frontend_only:
        entrypoint = tmp_path / "frontend" / "app.py"
        entrypoint.parent.mkdir()
        shutil.copyfile(source, entrypoint)
        assert not (entrypoint.parents[1] / "app").exists()
    else:
        entrypoint = source
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-I",
            "-c",
            """
import importlib.abc
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

entrypoint = Path(sys.argv[1])

class ForbidBackendImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"app", "torch", "transformers", "spacy"}:
            raise AssertionError(f"Frontend imported a backend module: {fullname}")

sys.meta_path.insert(0, ForbidBackendImports())
os.environ["API_URL"] = "http://backend:8000/"

app = AppTest.from_file(str(entrypoint))
for _ in range(2):
    app.run(timeout=20)
    assert not app.exception, [error.message for error in app.exception]
    assert app.title[0].value == "🎭 Text Style & Tone Detection"
    assert app.session_state["single_text"] == ""
    assert "app" not in sys.modules
    assert "torch" not in sys.modules
""",
            str(entrypoint),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
