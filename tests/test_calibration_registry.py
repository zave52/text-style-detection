"""Calibration, integrity, offline loading, and active-model regression tests."""

import json
import shutil
import socket

import pandas as pd
import pytest
import torch
from torch import nn

import app.experiments as experiments
from app.bundles import read_json, sha256, verify_bundle
from app.calibration import fit_temperature, fit_temperatures, temperatures_from
from app.config import STYLES, TONES
from app.experiments import TrainConfig, run_experiment
from app.prediction import NeuralPredictor
from app.registry import ActiveModelRegistry
from app.tokenization import Vocabulary, make_loader
from app.training import model_inputs


def frames():
    frame = pd.DataFrame(
        {
            "text": [
                "Hello world please send report now!",
                "Hello world!",
                "please send report",
            ],
            "style": ["business", "informal", "formal"],
            "tone": ["urgent", "friendly", "neutral"],
        }
    )
    return {
        split: frame.copy()
        for split in ("train", "validation", "test")
    }


def test_scalar_fit_has_known_optimum_and_preserves_predictions():
    logits = torch.tensor([[4.0, 0.0]] * 4)
    labels = torch.tensor([0, 0, 0, 1])

    result = fit_temperature(logits, labels)

    assert result["temperature"] == pytest.approx(
        4 / torch.log(torch.tensor(3.0)).item()
    )
    assert result["nll_after"] < result["nll_before"]
    assert torch.equal(
        logits.argmax(1),
        (logits / result["temperature"]).argmax(1),
    )

    sharper = fit_temperature(
        logits,
        torch.zeros(4, dtype=torch.long),
    )
    assert sharper["temperature"] == pytest.approx(0.05)

    flat = fit_temperature(torch.zeros_like(logits), labels)
    assert flat["temperature"] == 1


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_temperatures_are_rejected(bad):
    with pytest.raises(ValueError, match="strictly positive"):
        temperatures_from(
            {
                "method": "temperature_scaling",
                "split": "validation",
                "style": {"temperature": bad},
                "tone": {"temperature": 1.0},
            }
        )


def test_calibration_uses_eval_mode_and_never_updates_model_or_gradients():
    frame = frames()["validation"]
    vocabulary = Vocabulary.fit(frame["text"])

    class Frozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.arange(5, dtype=torch.float))
            self.seen = 0

        def forward(self, input_ids, lengths):
            assert not self.training and not torch.is_grad_enabled()
            self.seen += input_ids.shape[0]
            logits = self.bias.expand(input_ids.shape[0], -1)
            return {
                "style_logits": logits,
                "tone_logits": logits * 0.2,
            }

    model = Frozen().train()
    before = model.bias.detach().clone()
    model.bias.grad = torch.ones_like(model.bias)

    result = fit_temperatures(
        model,
        frame,
        vocabulary,
        batch_size=2,
    )

    assert result["split"] == "validation"
    assert result["samples"] == len(frame)
    assert model.training
    assert model.seen == len(frame)
    assert torch.equal(model.bias, before)
    assert torch.equal(model.bias.grad, torch.ones(5))
    assert result["style"]["temperature"] != result["tone"]["temperature"]

    for task in ("style", "tone"):
        assert result[task]["nll_after"] <= result[task]["nll_before"]


@pytest.fixture
def local_encoder(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    words = (
        "[PAD] [UNK] [CLS] [SEP] [MASK] "
        "Hello hello world please send report now !"
    ).split()

    backend = Tokenizer(
        WordPiece(
            dict(zip(words, range(len(words)), strict=True)),
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )

    source = tmp_path / "encoder"

    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=32,
    ).save_pretrained(source)

    DistilBertModel(
        DistilBertConfig(
            vocab_size=len(words),
            dim=8,
            hidden_dim=16,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=32,
        )
    ).save_pretrained(source)

    return source


@pytest.fixture
def calibrated_runs(tmp_path, monkeypatch, local_encoder):
    data = frames()

    monkeypatch.setattr(
        experiments,
        "load_splits",
        lambda: (data, {"seed": 42}),
    )

    paths = {}

    architectures = {
        "textcnn": {
            "embedding_dim": 4,
            "num_filters": 2,
        },
        "bilstm": {
            "embedding_dim": 4,
            "hidden_size": 3,
            "attention_dim": 2,
        },
        "distilbert": {
            "pretrained_name": str(local_encoder),
            "local_only": True,
        },
    }

    for name, kwargs in architectures.items():
        root = tmp_path / name

        run_experiment(
            TrainConfig(
                name,
                str(root),
                epochs=1,
                batch_size=2,
                max_length=16,
                model_kwargs=kwargs,
            )
        )
        paths[name] = root

    return paths


def test_all_architectures_preserve_logits_probabilities_labels_and_offline_loading(
    calibrated_runs,
    local_encoder,
    tmp_path,
    monkeypatch,
):
    from transformers import DistilBertModel

    texts = frames()["validation"]["text"].tolist()
    registry_path = tmp_path / "active.json"
    registry = ActiveModelRegistry(registry_path)
    expected = {}

    for name, root in calibrated_runs.items():
        manifest = verify_bundle(root)
        checkpoint = torch.load(root / "best.pt", weights_only=True)

        assert manifest["weights_sha256"] == sha256(root / "best.pt")
        assert manifest["dataset_manifest_sha256"] == sha256(
            root / "data_manifest.json"
        )
        assert checkpoint["calibration"]["samples"] == 3
        assert checkpoint["calibration"] == read_json(root / "calibration.json")
        assert read_json(root / "labels.json") == {
            "styles": list(STYLES),
            "tones": list(TONES),
        }

        predictor = NeuralPredictor(root, device="cpu")
        expected[name] = predictor.predict_proba(texts, batch_size=2)
        batch = next(iter(make_loader(texts, predictor.tokenizer)))

        with torch.no_grad():
            before = predictor.model(**model_inputs(batch))

        registry.select(root)
        selected = NeuralPredictor(
            registry_path=registry_path,
            device="cpu",
        )

        with torch.no_grad():
            after = selected.model(**model_inputs(batch))

        for key in before:
            torch.testing.assert_close(
                before[key],
                after[key],
                atol=0,
                rtol=0,
            )

        assert selected.predict_proba(texts, batch_size=2) == expected[name]
        assert selected.predict(texts) == predictor.predict(texts)

        for task, labels in (("style", STYLES), ("tone", TONES)):
            probabilities = torch.softmax(
                before[f"{task}_logits"].double() / selected.temperatures[task],
                dim=1,
            )

            assert list(expected[name][0][task].values()) == pytest.approx(
                probabilities[0].tolist()
            )
            assert selected.predict(texts)[0][task] == labels[
                before[f"{task}_logits"].argmax(1)[0]
            ]

    moved = tmp_path / "moved"
    shutil.copytree(calibrated_runs["distilbert"], moved)
    shutil.rmtree(local_encoder)

    for root in calibrated_runs.values():
        shutil.rmtree(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference attempted a network download.")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty_cache"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(DistilBertModel, "from_pretrained", forbidden)

    registry.select(moved, name="offline")

    assert (
        NeuralPredictor(registry_path=registry_path).predict_proba(texts, 2)
        == expected["distilbert"]
    )


@pytest.mark.parametrize(
    "filename",
    [
        "best.pt",
        "calibration.json",
        "vocabulary.json",
        "labels.json",
        "model_config.json",
        "training_metadata.json",
        "data_manifest.json",
    ],
)
def test_corrupted_files_are_rejected_before_loading(
    calibrated_runs,
    monkeypatch,
    filename,
):
    root = calibrated_runs["textcnn"]

    with (root / filename).open("ab") as stream:
        stream.write(b"modified")

    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail("Loaded before checking SHA-256"),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        NeuralPredictor(root)


def test_failed_selection_keeps_previous_active_model_and_pins_manifest(
    calibrated_runs,
    tmp_path,
):
    registry = ActiveModelRegistry(tmp_path / "active.json")
    good = calibrated_runs["bilstm"]
    bad = calibrated_runs["textcnn"]

    registry.select(good)
    before = registry.path.read_bytes()
    (bad / "labels.json").unlink()

    with pytest.raises(ValueError):
        registry.select(bad)

    assert registry.path.read_bytes() == before
    assert registry.resolve() == good

    bundle = read_json(good / "bundle.json")
    bundle["extra"] = "changed"

    (good / "bundle.json").write_text(
        json.dumps(bundle),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest checksum"):
        NeuralPredictor(registry_path=registry.path)


def test_missing_integrity_manifest_cannot_downgrade_calibrated_checkpoint(
    calibrated_runs,
):
    root = calibrated_runs["textcnn"]
    (root / "bundle.json").unlink()
    (root / "calibration.json").unlink()

    with pytest.raises(ValueError, match="verified bundle"):
        NeuralPredictor(root)


def test_training_never_passes_train_or_test_rows_to_calibration(
    tmp_path,
    monkeypatch,
):
    data = frames()
    data["train"]["text"] = "TRAIN ONLY example with enough tokens"
    data["test"]["text"] = "TEST ONLY must never calibrate"

    monkeypatch.setattr(
        experiments,
        "load_splits",
        lambda: (data, {"seed": 42}),
    )

    calls = []
    original = experiments.fit_temperatures

    def capture(model, validation_data, tokenizer, **kwargs):
        assert validation_data is data["validation"]
        calls.append(validation_data["text"].tolist())
        assert not model.training
        return original(model, validation_data, tokenizer, **kwargs)

    monkeypatch.setattr(experiments, "fit_temperatures", capture)

    run_experiment(
        TrainConfig(
            "textcnn",
            str(tmp_path / "run"),
            epochs=1,
            evaluate_test=True,
            model_kwargs={
                "embedding_dim": 4,
                "num_filters": 2,
            },
        )
    )

    assert calls == [data["validation"]["text"].tolist()]


def test_calibration_cli_reuses_weights_and_only_encodes_validation(
    calibrated_runs,
    monkeypatch,
):
    import scripts.calibrate as cli

    root = calibrated_runs["textcnn"]
    checkpoint = torch.load(root / "best.pt", weights_only=True)
    before = {
        key: value.clone()
        for key, value in checkpoint["state_dict"].items()
    }

    checkpoint.pop("calibration")
    torch.save(checkpoint, root / "best.pt")
    (root / "bundle.json").unlink()
    (root / "calibration.json").unlink()

    data = frames()
    data["train"] = data["test"] = pd.DataFrame({"text": [None]})

    monkeypatch.setattr(
        cli,
        "load_splits",
        lambda: (data, {"seed": 42}),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "calibrate",
            "--checkpoint",
            str(root),
            "--device",
            "cpu",
        ],
    )

    cli.main()
    verify_bundle(root)

    after = torch.load(root / "best.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(before[key], after[key]) for key in before)

    with pytest.raises(FileExistsError, match="already calibrated"):
        cli.main()


def test_calibration_cli_rejects_changed_dataset_before_loading_model(
    calibrated_runs,
    monkeypatch,
):
    import scripts.calibrate as cli

    root = calibrated_runs["textcnn"]
    (root / "bundle.json").unlink()

    monkeypatch.setattr(
        cli,
        "load_splits",
        lambda: (frames(), {"seed": 99}),
    )
    monkeypatch.setattr(
        cli,
        "NeuralPredictor",
        lambda *a, **k: pytest.fail("Loaded wrong dataset"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "--checkpoint", str(root)],
    )

    with pytest.raises(ValueError, match="saved manifest"):
        cli.main()


def test_selection_cli_switches_architectures(
    calibrated_runs,
    tmp_path,
    monkeypatch,
):
    import scripts.select_model as cli

    registry_path = tmp_path / "registry.json"

    for architecture in ("textcnn", "bilstm", "distilbert"):
        monkeypatch.setattr(
            "sys.argv",
            [
                "select_model",
                "--checkpoint",
                str(calibrated_runs[architecture]),
                "--registry",
                str(registry_path),
            ],
        )

        cli.main()

        assert (
            NeuralPredictor(registry_path=registry_path).model_name
            == architecture
        )
