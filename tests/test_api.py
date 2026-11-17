from fastapi.testclient import TestClient

from app.main import create_app


def prediction(text):
    return {
        "model": "textcnn",
        "style": "formal",
        "tone": "neutral",
        "style_confidence": 0.8,
        "tone_confidence": 0.8,
        "style_scores": {"formal": 0.8, "casual": 0.2},
        "tone_scores": {"neutral": 0.8, "urgent": 0.2},
        "character_count": len(text),
        "token_count": 4,
        "used_token_count": 4,
        "max_length": 128,
        "truncated": False,
        "unknown_token_count": 0,
        "unknown_ratio": 0.0,
        "warnings": [],
    }


class StubPredictionService:
    def predict(self, text):
        return prediction(text)

    def predict_batch(self, texts):
        return [prediction(text) for text in texts]

    def info(self):
        return {"model": "textcnn"}


def test_prediction_routes_return_the_shared_response_contract():
    with TestClient(create_app(StubPredictionService())) as client:
        responses = [
            client.post(endpoint, json={"text": "Please review this report."})
            for endpoint in ("/predict", "/predict/style", "/predict/tone")
        ]
        batch = client.post("/predict/batch", json={"texts": ["Review this report."]})

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == responses[1].json() == responses[2].json()
    assert batch.status_code == 200
    assert batch.json() == {"results": [prediction("Review this report.")]}


def test_prediction_routes_reject_invalid_requests():
    with TestClient(create_app(StubPredictionService())) as client:
        missing_text = client.post("/predict", json={})
        blank_text = client.post("/predict", json={"text": "  "})
        empty_batch = client.post("/predict/batch", json={"texts": []})

    assert missing_text.status_code == 422
    assert blank_text.status_code == 422
    assert empty_batch.status_code == 422
