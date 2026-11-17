import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from streamlit.testing.v1 import AppTest


def run_app(monkeypatch):
    monkeypatch.setenv("API_URL", "http://api.test")
    script = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    return AppTest.from_file(script).run()


def test_interface_renders_prediction_controls(monkeypatch):
    app = run_app(monkeypatch)

    assert not app.exception
    assert len(app.text_area) == 1
    assert [button.label for button in app.button] == [
        "Predict Both (Style & Tone)",
        "Style Only",
        "Tone Only",
    ]


def test_empty_input_does_not_call_api(monkeypatch):
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    app = run_app(monkeypatch)
    app.button[0].click().run()

    post.assert_not_called()
    assert any("Please enter some text first." in warning.value for warning in app.warning)


def test_successful_prediction_is_displayed(monkeypatch):
    post = Mock(
        return_value=Mock(status_code=200, json=lambda: {"style": "formal", "tone": "neutral"})
    )
    monkeypatch.setattr(requests, "post", post)
    app = run_app(monkeypatch)
    app.text_area[0].set_value("Please review this report.").run()
    app.button[0].click().run()

    post.assert_called_once_with(
        "http://api.test/predict",
        json={"text": "Please review this report."},
    )
    assert any("formal" in item.value for item in app.markdown)
    assert any("neutral" in item.value for item in app.markdown)


def test_api_error_is_shown_to_user(monkeypatch):
    post = Mock(return_value=Mock(status_code=503, text="Model unavailable"))
    monkeypatch.setattr(requests, "post", post)
    app = run_app(monkeypatch)
    app.text_area[0].set_value("Please review this report.").run()
    app.button[0].click().run()

    assert any("Error 503: Model unavailable" in error.value for error in app.error)


def test_connection_error_is_shown_to_user(monkeypatch):
    monkeypatch.setattr(requests, "post", Mock(side_effect=requests.ConnectionError))
    app = run_app(monkeypatch)
    app.text_area[0].set_value("Please review this report.").run()
    app.button[0].click().run()

    assert any("Error connecting to the backend" in error.value for error in app.error)


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
