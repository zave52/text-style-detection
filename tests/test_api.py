import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import MAX_BATCH_SIZE, MAX_INPUT_CHARS, STYLES, TONES
from app.input_validation import PredictionInputError
from app.main import BATCH_PREDICTION_FAILED, MODEL_UNAVAILABLE, PREDICTION_FAILED, create_app


def prediction(text):
    return {
        "model": "textcnn",
        "style": "formal",
        "tone": "neutral",
        "style_confidence": 0.8,
        "tone_confidence": 0.8,
        "style_scores": {"formal": 0.8, "informal": 0.2},
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


MODEL_INFO = {
    "model": "textcnn",
    "language": "en",
    "styles": list(STYLES),
    "tones": list(TONES),
    "max_length": 128,
    "tokenizer_kind": "vocabulary",
    "device": "cpu",
    "temperatures": {"style": 1.2, "tone": 0.9},
    "weights_sha256": "0" * 64,
    "input_limits": {"min_characters": 1, "max_characters": 2000, "max_batch_size": 64},
    "warning_thresholds": {"confidence": 0.55, "margin": 0.15, "unknown_ratio": 0.3},
}


class StubPredictionService:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return prediction(text)

    def predict_batch(self, texts):
        self.calls.append(texts)
        if self.error:
            raise self.error
        return [prediction(text) for text in texts]

    def info(self):
        return MODEL_INFO


@pytest.fixture
def stub():
    return StubPredictionService()


@pytest.fixture
def client(stub):
    with TestClient(create_app(stub)) as test_client:
        yield test_client


@pytest.fixture
def unavailable_client(monkeypatch):
    def failing_loader():
        raise FileNotFoundError("artifacts/active_model.json")

    failing_loader.cache_clear = lambda: None
    monkeypatch.setattr(main, "get_prediction_service", failing_loader)
    with TestClient(create_app()) as test_client:
        yield test_client


def test_prediction_routes_return_the_shared_response_contract(client):
    responses = [
        client.post(endpoint, json={"text": "Please review this report."})
        for endpoint in ("/predict", "/predict/style", "/predict/tone")
    ]

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == prediction("Please review this report.")
    assert responses[0].json() == responses[1].json() == responses[2].json()


def test_batch_preserves_order_and_duplicates(client, stub):
    texts = ["Review this report.", "Hello!", "Review this report."]
    response = client.post("/predict/batch", json={"texts": texts})

    assert response.status_code == 200
    assert response.json() == {"results": [prediction(text) for text in texts]}
    assert stub.calls == [texts]


def test_text_is_passed_to_the_service_unchanged(client, stub):
    client.post("/predict", json={"text": "  Keep MY  spacing!  "})

    assert stub.calls == ["  Keep MY  spacing!  "]


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        ("/predict", {}),
        ("/predict", {"text": "  "}),
        ("/predict", {"text": ""}),
        ("/predict", {"text": 42}),
        ("/predict", {"text": "x" * (MAX_INPUT_CHARS + 1)}),
        ("/predict", {"text": "Hello", "extra": True}),
        ("/predict/style", {"text": None}),
        ("/predict/tone", {"texts": ["Hello"]}),
        ("/predict/batch", {"texts": []}),
        ("/predict/batch", {"texts": "Hello"}),
        ("/predict/batch", {"texts": ["Hello", " "]}),
        ("/predict/batch", {"texts": ["Hello", 1]}),
        ("/predict/batch", {"texts": ["Hi"] * (MAX_BATCH_SIZE + 1)}),
    ],
)
def test_invalid_requests_are_rejected_before_inference(client, stub, endpoint, body):
    response = client.post(endpoint, json=body)

    assert response.status_code == 422
    assert "detail" in response.json()
    assert stub.calls == []


def test_largest_valid_inputs_are_accepted(client):
    single = client.post("/predict", json={"text": "x" * MAX_INPUT_CHARS})
    batch = client.post("/predict/batch", json={"texts": ["Hi"] * MAX_BATCH_SIZE})

    assert single.status_code == 200
    assert len(batch.json()["results"]) == MAX_BATCH_SIZE


def test_service_input_errors_are_reported_as_422():
    service = StubPredictionService(PredictionInputError("Message at index 0: too long."))
    with TestClient(create_app(service)) as client:
        response = client.post("/predict", json={"text": "Hello"})

    assert response.status_code == 422
    assert response.json() == {"detail": "Message at index 0: too long."}


def test_inference_failures_return_503_without_internal_details():
    service = StubPredictionService(RuntimeError("CUDA out of memory at /secret/path"))
    with TestClient(create_app(service)) as client:
        single = client.post("/predict", json={"text": "Hello"})
        batch = client.post("/predict/batch", json={"texts": ["Hello"]})

    assert single.status_code == batch.status_code == 503
    assert single.json() == {"detail": PREDICTION_FAILED}
    assert batch.json() == {"detail": BATCH_PREDICTION_FAILED}


def test_health_and_model_info_when_model_is_loaded(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    ready = client.get("/health/ready")
    info = client.get("/model/info")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model": "textcnn"}
    assert info.status_code == 200
    assert info.json() == MODEL_INFO


def test_loading_failure_keeps_process_live_but_not_ready(unavailable_client):
    assert unavailable_client.get("/health/live").status_code == 200
    blocked = [
        unavailable_client.get("/health/ready"),
        unavailable_client.get("/model/info"),
        unavailable_client.post("/predict", json={"text": "Hello"}),
        unavailable_client.post("/predict/style", json={"text": "Hello"}),
        unavailable_client.post("/predict/tone", json={"text": "Hello"}),
        unavailable_client.post("/predict/batch", json={"texts": ["Hello"]}),
    ]

    assert all(response.status_code == 503 for response in blocked)
    assert all(response.json() == {"detail": MODEL_UNAVAILABLE} for response in blocked)
    assert "active_model.json" not in str([response.json() for response in blocked])


def test_openapi_documents_every_endpoint_and_error(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    expected = {
        "/predict": "post",
        "/predict/style": "post",
        "/predict/tone": "post",
        "/predict/batch": "post",
        "/model/info": "get",
        "/health/live": "get",
        "/health/ready": "get",
    }
    assert expected.items() <= {path: next(iter(ops)) for path, ops in paths.items()}.items()
    for path in ("/predict", "/predict/style", "/predict/tone", "/predict/batch"):
        assert {"200", "422", "503"} <= set(paths[path]["post"]["responses"])
    assert "503" in paths["/health/ready"]["get"]["responses"]
    assert {"TextRequest", "BatchRequest", "PredictResponse", "ModelInfoResponse"} <= set(
        schema["components"]["schemas"]
    )
    operation_ids = [op["operationId"] for ops in paths.values() for op in ops.values()]
    assert len(operation_ids) == len(set(operation_ids))
