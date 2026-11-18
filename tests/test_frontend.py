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


def prediction_response():
    return {
        "model": "textcnn",
        "style": "formal",
        "tone": "neutral",
        "style_confidence": 0.8,
        "tone_confidence": 0.8,
        "style_scores": {"formal": 0.8, "casual": 0.2},
        "tone_scores": {"neutral": 0.8, "urgent": 0.2},
        "character_count": 26,
        "token_count": 4,
        "used_token_count": 4,
        "max_length": 128,
        "truncated": False,
        "unknown_token_count": 0,
        "unknown_ratio": 0.0,
        "warnings": [],
    }


def test_interface_renders_single_and_batch_controls(monkeypatch):
    app = run_app(monkeypatch)

    assert not app.exception
    assert [text_area.label for text_area in app.text_area] == ["Your text", "Messages"]
    assert {button.key for button in app.button} == {"both", "style", "tone", "batch"}


def test_prediction_is_sent_and_displayed(monkeypatch):
    response = Mock(ok=True, json=prediction_response)
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    app = run_app(monkeypatch)
    app.text_area(key="single_text").set_value("Please review this report.")
    app.button(key="both").click().run()

    post.assert_called_once_with(
        "http://api.test/predict",
        json={"text": "Please review this report."},
        timeout=(5, 120),
    )
    assert not app.exception
    assert any(metric.label == "Style" and metric.value == "formal" for metric in app.metric)
    assert any(metric.label == "Tone" and metric.value == "neutral" for metric in app.metric)


def test_api_validation_error_is_shown_to_user(monkeypatch):
    response = Mock(
        ok=False,
        json=lambda: {"detail": "Messages must not be empty or contain only whitespace."},
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    app = run_app(monkeypatch)
    app.button(key="both").click().run()

    post.assert_called_once()
    assert any("Messages must not be empty" in error.value for error in app.error)


def test_connection_error_is_shown_to_user(monkeypatch):
    monkeypatch.setattr(requests, "post", Mock(side_effect=requests.ConnectionError))
    app = run_app(monkeypatch)
    app.text_area(key="single_text").set_value("Please review this report.")
    app.button(key="both").click().run()

    assert any("Cannot reach the prediction API" in error.value for error in app.error)


@pytest.mark.parametrize("frontend_only", [False, True])
def test_streamlit_starts_without_backend_imports(tmp_path, frontend_only):
    source = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    if frontend_only:
        entrypoint = tmp_path / "frontend" / "app.py"
        entrypoint.parent.mkdir()
        shutil.copyfile(source, entrypoint)
        shutil.copyfile(source.with_name("batch.py"), entrypoint.with_name("batch.py"))
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
