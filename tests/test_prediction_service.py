"""Shared service, CLI, API, and UI prediction-contract regression tests."""

import json
import shutil
import socket
import stat
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.bundles import save_calibrated_bundle, sha256, write_json
from app.config import MAX_BATCH_SIZE, MAX_INPUT_CHARS, STYLES, TONES
from app.experiments import TrainConfig, save_best
from app.input_validation import PredictionInputError
from app.main import create_app
from app.models import build_model
from app.registry import ActiveModelRegistry
from app.service import PredictionService, get_prediction_service
from app.tokenization import TransformerTokenizer, Vocabulary


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("prediction_bundles")
    words = "[PAD] [UNK] [CLS] [SEP] [MASK] hello Hello world please send report now !".split()
    backend = Tokenizer(WordPiece({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer_dir = root / "tokenizer_source"
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=32,
    ).save_pretrained(tokenizer_dir)
    paths = {}
    for architecture in ("textcnn", "bilstm", "distilbert"):
        directory = root / architecture
        directory.mkdir()
        kwargs = {"num_styles": len(STYLES), "num_tones": len(TONES)}
        if architecture == "distilbert":
            from transformers import DistilBertConfig

            tokenizer = TransformerTokenizer(tokenizer_dir, max_length=8, local_only=True)
            kwargs["encoder_config"] = DistilBertConfig(
                vocab_size=len(words),
                dim=8,
                hidden_dim=16,
                n_heads=2,
                n_layers=1,
                max_position_embeddings=32,
            ).to_dict()
        else:
            tokenizer = Vocabulary.fit(["hello world please send report now ! HELLO"], max_length=8)
            kwargs.update(vocab_size=len(tokenizer), pad_id=tokenizer.pad_id, embedding_dim=4)
            kwargs.update(
                {"num_filters": 2}
                if architecture == "textcnn"
                else {
                    "hidden_size": 3,
                    "attention_dim": 2,
                }
            )
        model = build_model(architecture, **kwargs)
        with torch.no_grad():
            model.style_head.weight.zero_()
            model.tone_head.weight.zero_()
            model.style_head.bias.copy_(torch.arange(5) * 0.4)
            model.tone_head.bias.copy_(torch.arange(4, -1, -1) * 0.4)
        config = TrainConfig(architecture, str(directory), max_length=8)
        save_best(model, tokenizer, kwargs, config, directory, 1, {"mean_macro_f1": 0.2})
        for name, value in {
            "training_config.json": {"settings": asdict(config)},
            "data_manifest.json": {"seed": 42},
            "history.json": [],
            "summary.json": {},
        }.items():
            write_json(directory / name, value)
        save_calibrated_bundle(
            directory,
            {
                "method": "temperature_scaling",
                "split": "validation",
                "samples": 2,
                "style": {"temperature": 2.0},
                "tone": {"temperature": 3.0},
            },
        )
        paths[architecture] = directory
    return paths


@pytest.mark.parametrize("architecture", ["textcnn", "bilstm", "distilbert"])
def test_single_batch_and_calibration_are_consistent_offline(bundles, monkeypatch, architecture):
    from transformers import DistilBertModel

    def forbidden(*args, **kwargs):
        raise AssertionError("Prediction tried to download a base model")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(DistilBertModel, "from_pretrained", forbidden)
    service = PredictionService(bundles[architecture], device="cpu")
    texts = ["hello", "hello world!", "hello " * 20, "zzzzzz", "Привіт", "123 !!!"]
    batch = service.predict_batch(texts, batch_size=2)
    assert batch == [service.predict(text) for text in texts]
    for row in batch:
        assert row["model"] == architecture
        assert row["used_token_count"] <= row["max_length"] == 8
        for task, labels, temperature, bias in (
            ("style", STYLES, 2.0, torch.arange(5) * 0.4),
            ("tone", TONES, 3.0, torch.arange(4, -1, -1) * 0.4),
        ):
            assert tuple(row[f"{task}_scores"]) == labels
            expected = torch.softmax(bias.double() / temperature, dim=0)
            assert list(row[f"{task}_scores"].values()) == pytest.approx(expected.tolist())
            assert row[f"{task}_confidence"] == max(row[f"{task}_scores"].values())
            assert row[task] == labels[expected.argmax()]
    assert batch[2]["truncated"]
    assert any(w["code"] == "truncated" for w in batch[2]["warnings"])
    assert batch[3]["unknown_token_count"] == 1 and batch[3]["unknown_ratio"] == 1
    assert any(w["code"] == "unfamiliar_input" for w in batch[3]["warnings"])
    assert any(w["code"] == "unfamiliar_input" for w in batch[4]["warnings"])
    assert any(w["code"] == "unfamiliar_input" for w in batch[5]["warnings"])
    assert {w["task"] for w in batch[0]["warnings"] if w["code"] == "uncertain_prediction"} == {
        "style",
        "tone",
    }
    if architecture == "distilbert":
        assert batch[0]["token_count"] == batch[0]["used_token_count"] == 3
    assert service.info()["weights_sha256"] == sha256(bundles[architecture] / "best.pt")


@pytest.mark.parametrize(
    "texts",
    [
        [],
        "hello",
        [""],
        ["  \n\t"],
        [None],
        [123],
        [True],
        ["hello", ""],
        ["x" * (MAX_INPUT_CHARS + 1)],
        ["hello"] * (MAX_BATCH_SIZE + 1),
    ],
)
def test_invalid_requests_are_rejected_before_inference(bundles, monkeypatch, texts):
    service = PredictionService(bundles["textcnn"])
    monkeypatch.setattr(
        service.predictor, "predict_proba", lambda *a, **k: pytest.fail("Ran inference")
    )
    with pytest.raises(PredictionInputError):
        service.predict_batch(texts)


@pytest.mark.parametrize("batch_size", [0, 65, 2.5, True])
def test_invalid_internal_batch_size_is_rejected(bundles, batch_size):
    with pytest.raises(PredictionInputError):
        PredictionService(bundles["textcnn"]).predict_batch(["hello"], batch_size=batch_size)


def test_input_boundaries_and_threshold_validation(bundles):
    service = PredictionService(bundles["textcnn"])
    assert service.predict("x" * MAX_INPUT_CHARS)["character_count"] == MAX_INPUT_CHARS
    assert len(service.predict_batch(["hello"] * MAX_BATCH_SIZE)) == MAX_BATCH_SIZE
    for value in (-1, 2, float("nan"), True, "low"):
        with pytest.raises(ValueError):
            PredictionService(bundles["textcnn"], confidence_threshold=value)


def test_warning_thresholds_do_not_change_predictions_or_calibration(bundles):
    default = PredictionService(bundles["textcnn"]).predict("zzzzzz")
    quiet = PredictionService(
        bundles["textcnn"], confidence_threshold=0, margin_threshold=0, unknown_threshold=1
    ).predict("zzzzzz")
    assert default["warnings"] and not quiet["warnings"]
    assert {key: value for key, value in default.items() if key != "warnings"} == {
        key: value for key, value in quiet.items() if key != "warnings"
    }


@pytest.mark.parametrize(
    "filename",
    ["best.pt", "labels.json", "calibration.json", "vocabulary.json", "model_config.json"],
)
def test_corrupted_bundle_is_rejected_before_torch_load(bundles, tmp_path, monkeypatch, filename):
    root = tmp_path / "model"
    shutil.copytree(bundles["textcnn"], root)
    with (root / filename).open("ab") as stream:
        stream.write(b"modified")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Loaded before verification"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        PredictionService(root)


def test_registry_selection_and_cached_factory(bundles, tmp_path, monkeypatch):
    path = tmp_path / "active.json"
    registry = ActiveModelRegistry(path)
    registry.select(bundles["bilstm"])
    monkeypatch.setenv("TSD_REGISTRY", str(path))
    monkeypatch.delenv("TSD_CHECKPOINT", raising=False)
    get_prediction_service.cache_clear()
    try:
        first = get_prediction_service()
        assert first is get_prediction_service()
        assert first.info()["model"] == "bilstm"
        registry.select(bundles["textcnn"])
        assert first.info()["model"] == "bilstm"
        get_prediction_service.cache_clear()
        assert get_prediction_service().info()["model"] == "textcnn"
    finally:
        get_prediction_service.cache_clear()


def test_cli_api_and_direct_service_return_identical_fields(bundles, tmp_path, monkeypatch, capsys):
    import app.prediction as cli

    registry_path = tmp_path / "active.json"
    ActiveModelRegistry(registry_path).select(bundles["textcnn"])
    service = PredictionService(registry_path=registry_path)
    texts = ["hello", "zzzzzz"]
    expected = service.predict_batch(texts)
    monkeypatch.setattr("sys.argv", ["prediction", "--registry", str(registry_path), *texts])
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"results": expected}
    monkeypatch.setattr("sys.argv", ["prediction", "--registry", str(registry_path), "hello"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == expected[0]
    with TestClient(create_app(service)) as client:
        for endpoint in ("/predict", "/predict/style", "/predict/tone"):
            response = client.post(endpoint, json={"text": texts[0]})
            assert response.status_code == 200 and response.json() == expected[0]
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 200 and response.json() == {"results": expected}
        assert client.get("/health/ready").status_code == 200
        assert client.get("/model/info").json() == service.info()
        for body in ({"text": " "}, {"text": 1}, {"text": "x" * 2001}):
            assert client.post("/predict", json=body).status_code == 422
        for body in ({"texts": []}, {"texts": ["hello", ""]}, {"texts": ["hello"] * 65}):
            assert client.post("/predict/batch", json=body).status_code == 422


def test_unavailable_model_returns_readiness_503_without_exposing_paths(monkeypatch):
    import app.main as api

    def unavailable():
        raise FileNotFoundError("private/path/to/model.pt")

    monkeypatch.setattr(get_prediction_service, "__wrapped__", unavailable)
    monkeypatch.setattr(api, "get_prediction_service", unavailable)
    unavailable.cache_clear = lambda: None
    with TestClient(api.create_app()) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        response = client.post("/predict", json={"text": "hello"})
        assert response.status_code == 503
        assert "private/path" not in response.text


def test_api_loads_the_checkpoint_once_per_lifespan(bundles, monkeypatch):
    monkeypatch.setenv("TSD_CHECKPOINT", str(bundles["textcnn"]))
    get_prediction_service.cache_clear()
    original = torch.load
    loads = []

    def counted(*args, **kwargs):
        loads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.post("/predict", json={"text": "hello"}).status_code == 200
        assert client.post("/predict/batch", json={"texts": ["hello", "world"]}).status_code == 200
        assert len(loads) == 1


def test_invalid_model_output_returns_503(bundles, monkeypatch):
    service = PredictionService(bundles["textcnn"])
    scores = service.predictor.predict_proba(["hello"])
    scores[0]["style"][STYLES[0]] = float("nan")
    monkeypatch.setattr(service.predictor, "predict_proba", lambda *a, **k: scores)
    with TestClient(create_app(service)) as client:
        response = client.post("/predict", json={"text": "hello"})
        assert response.status_code == 503
        assert "nan" not in response.text


def test_streamlit_uses_the_same_response_for_single_and_batch(bundles, monkeypatch):
    import requests
    from streamlit.testing.v1 import AppTest

    service = PredictionService(bundles["textcnn"])
    calls = []
    with TestClient(create_app(service)) as client:

        def post(url, *, json, timeout):
            calls.append((url, json))
            endpoint = "/" + url.split(":8000/", 1)[1]
            result = client.post(endpoint, json=json)
            response = requests.Response()
            response.status_code = result.status_code
            response._content = result.content
            return response

        monkeypatch.setattr(requests, "post", post)
        app = AppTest.from_file(str(Path(__file__).parents[1] / "frontend" / "app.py"))
        app.run()
        assert not app.exception
        app.text_area(key="single_text").set_value("hello")
        app.button(key="both").click().run()
        assert not app.exception
        assert json.loads(app.json[0].value) == service.predict("hello")
        app.text_area(key="batch_text").set_value("hello\nzzzzzz")
        app.button(key="batch").click().run()
        assert not app.exception
        assert json.loads(app.json[0].value) == {
            "results": service.predict_batch(["hello", "zzzzzz"])
        }
        assert len(calls) == 2

        for invalid in (" ", "x" * (MAX_INPUT_CHARS + 1)):
            app.text_area(key="single_text").set_value(invalid)
            app.button(key="both").click().run()
            assert not app.exception
            assert calls[-1][1] == {"text": invalid}
            assert app.error

        app.text_area(key="batch_text").set_value("\n".join(["hello"] * (MAX_BATCH_SIZE + 1)))
        app.button(key="batch").click().run()
        assert not app.exception
        assert len(calls[-1][1]["texts"]) == MAX_BATCH_SIZE + 1
        assert app.error


@pytest.mark.parametrize("architecture", ["textcnn", "bilstm", "distilbert"])
def test_api_loads_selected_bundle_from_read_only_artifacts(
    bundles, tmp_path, monkeypatch, architecture
):
    artifacts = tmp_path / "artifacts"
    root = artifacts / "experiments" / architecture
    shutil.copytree(bundles[architecture], root)
    registry = artifacts / "active_model.json"
    ActiveModelRegistry(registry).select(root)
    original = PredictionService(registry_path=registry, device="cpu").predict("hello")
    files = [path for path in artifacts.rglob("*") if path.is_file()]
    entries = set(artifacts.rglob("*"))
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in [artifacts, *entries]}
    checksums = {path: sha256(path) for path in files}
    monkeypatch.delenv("TSD_CHECKPOINT", raising=False)
    monkeypatch.setenv("TSD_REGISTRY", str(registry))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    get_prediction_service.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not download a base model or tokenizer.")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    try:
        for path in modes:
            path.chmod(0o555 if path.is_dir() else 0o444)
        with TestClient(create_app()) as client:
            assert client.get("/health/ready").json() == {
                "status": "ready",
                "model": architecture,
            }
            response = client.post("/predict", json={"text": "hello"})
            assert response.status_code == 200
            assert response.json() == original
        assert {path: sha256(path) for path in files} == checksums
        assert set(artifacts.rglob("*")) == entries
    finally:
        for path, mode in modes.items():
            path.chmod(mode)
