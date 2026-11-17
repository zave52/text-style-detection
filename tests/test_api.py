from fastapi.testclient import TestClient

from app import main
from app.ml import models


def make_client(monkeypatch):
    monkeypatch.setattr(main, "load_models", lambda: None)
    monkeypatch.setattr(main, "clear_models", lambda: None)
    return TestClient(main.app)


def test_predict_returns_both_labels(monkeypatch):
    monkeypatch.setattr(main, "predict_both", lambda text: ("formal", "neutral"))

    with make_client(monkeypatch) as client:
        response = client.post("/predict", json={"text": "Please review this report."})

    assert response.status_code == 200
    assert response.json() == {"style": "formal", "tone": "neutral"}


def test_predict_style_returns_style_only(monkeypatch):
    monkeypatch.setattr(main, "predict_style", lambda text: "academic")

    with make_client(monkeypatch) as client:
        response = client.post("/predict/style", json={"text": "The results are significant."})

    assert response.status_code == 200
    assert response.json() == {"style": "academic"}


def test_predict_tone_returns_tone_only(monkeypatch):
    monkeypatch.setattr(main, "predict_tone", lambda text: "urgent")

    with make_client(monkeypatch) as client:
        response = client.post("/predict/tone", json={"text": "Send this immediately."})

    assert response.status_code == 200
    assert response.json() == {"tone": "urgent"}


def test_predict_rejects_missing_text(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_reports_unavailable_model(monkeypatch):
    from fastapi import HTTPException

    def unavailable(text):
        raise HTTPException(status_code=503, detail="Style model is not loaded.")

    monkeypatch.setattr(main, "predict_both", unavailable)

    with make_client(monkeypatch) as client:
        response = client.post("/predict", json={"text": "A valid input."})

    assert response.status_code == 503
    assert response.json() == {"detail": "Style model is not loaded."}


def test_lifespan_clears_loaded_models(monkeypatch):
    cleared = []

    def load_models():
        models["style"] = object()

    def clear_models():
        cleared.append(True)
        models.clear()

    monkeypatch.setattr(main, "load_models", load_models)
    monkeypatch.setattr(main, "clear_models", clear_models)

    with TestClient(main.app):
        assert "style" in models

    assert cleared == [True]
    assert models == {}
