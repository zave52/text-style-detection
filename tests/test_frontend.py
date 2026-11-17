from pathlib import Path
from unittest.mock import Mock

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
