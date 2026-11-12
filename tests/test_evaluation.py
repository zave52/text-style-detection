"""Held-out evaluation metrics, compatibility checks, and export regression tests."""

import shutil
import socket

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.metrics import f1_score

import app.experiments as experiments
from app.bundles import read_json, sha256, write_json
from app.config import STYLES, TONES
from app.data import load_splits
from app.evaluation import evaluate_saved_models, probability_metrics
from app.evaluation_reports import business_formal_stats, export_model_reports, recurring_errors
from app.experiments import TrainConfig, run_experiment
from app.prediction import NeuralPredictor


@pytest.fixture(scope="module")
def saved_models(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("evaluation_models")
    data = root / "data"
    data.mkdir()
    for split in ("train", "validation", "test"):
        pd.DataFrame(
            {
                "sample_id": [f"{split}-{index}" for index in range(5)],
                "group_id": [f"group-{split}-{index}" for index in range(5)],
                "text": [f"{split} Hello world please send report {index}!" for index in range(5)],
                "style": list(STYLES),
                "tone": list(TONES),
            }
        ).to_csv(data / f"{split}.csv", index=False)
    manifest = {
        "source_sha256": "a" * 64,
        "splits": {
            split: {"rows": 5, "sha256": sha256(data / f"{split}.csv")}
            for split in ("train", "validation", "test")
        },
    }
    write_json(data / "manifest.json", manifest)
    words = (
        "[PAD] [UNK] [CLS] [SEP] [MASK] train validation test Hello world please send report !"
    ).split()
    backend = Tokenizer(WordPiece({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    encoder = root / "base"
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=32,
    ).save_pretrained(encoder)
    DistilBertModel(
        DistilBertConfig(
            vocab_size=len(words),
            dim=8,
            hidden_dim=16,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=32,
        )
    ).save_pretrained(encoder)
    paths = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(experiments, "load_splits", lambda: load_splits(data))
        for name, kwargs in {
            "textcnn": {"embedding_dim": 4, "num_filters": 2},
            "bilstm": {"embedding_dim": 4, "hidden_size": 3, "attention_dim": 2},
            "distilbert": {"pretrained_name": str(encoder), "local_only": True},
        }.items():
            paths.append(
                run_experiment(
                    TrainConfig(
                        name,
                        str(root / name),
                        epochs=1,
                        batch_size=2,
                        max_length=8,
                        model_kwargs=kwargs,
                    )
                )
            )
    return data, paths


def test_probability_metrics_match_hand_calculation_and_include_confidence_one():
    logits = np.log([[0.8, 0.2], [0.4, 0.6]])
    metrics, rows = probability_metrics(logits, [0, 1], 1.0, 2)
    assert metrics["nll"] == pytest.approx(-np.log([0.8, 0.6]).mean())
    assert metrics["brier"] == pytest.approx(0.2)
    assert metrics["ece"] == pytest.approx(0.3)
    assert sum(row["samples"] for row in rows) == 2
    extreme, rows = probability_metrics([[1000, -1000], [-1000, 1000]], [1, 1], 1.0, 15)
    assert extreme == pytest.approx({"nll": 1000, "brier": 1, "ece": 0.5})
    assert rows[-1]["samples"] == 2


def test_all_architectures_use_same_test_rows_and_export_correct_metrics(
    saved_models, tmp_path, monkeypatch
):
    from transformers import DistilBertModel

    import app.calibration as calibration
    import scripts.evaluate as cli

    data, paths = saved_models
    before = {path: sha256(path / "best.pt") for path in paths}

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not fit or download models")

    monkeypatch.setattr(calibration, "fit_temperature", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(DistilBertModel, "from_pretrained", forbidden)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    output = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--checkpoints",
            *map(str, paths),
            "--data-dir",
            str(data),
            "--output",
            str(output),
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    cli.main()
    comparison = pd.read_csv(output / "comparison.csv")
    assert set(comparison["model_name"]) == {"textcnn", "bilstm", "distilbert"}
    test = load_splits(data)[0]["test"]
    assert read_json(output / "evaluation_config.json")["test_sha256"] == sha256(data / "test.csv")
    for row, path in zip(comparison.itertuples(), paths, strict=True):
        root = output / row.run_id
        predictions = pd.read_csv(root / "predictions.csv")
        assert predictions["sample_id"].tolist() == test["sample_id"].tolist()
        assert predictions["text"].tolist() == test["text"].tolist()
        scores = read_json(root / "metrics.json")["metrics"]
        assert scores["joint_accuracy"] == pytest.approx(predictions["joint_correct"].mean())
        predictor = NeuralPredictor(path, device="cpu")
        expected = predictor.predict_proba(test["text"].tolist(), batch_size=2)
        for task, labels in (("style", STYLES), ("tone", TONES)):
            assert scores[task]["accuracy"] == pytest.approx(predictions[f"{task}_correct"].mean())
            assert scores[task]["weighted_f1"] == pytest.approx(
                f1_score(
                    test[task],
                    predictions[f"predicted_{task}"],
                    labels=list(labels),
                    average="weighted",
                    zero_division=0,
                )
            )
            assert scores[task]["macro_f1"] == pytest.approx(
                f1_score(
                    test[task],
                    predictions[f"predicted_{task}"],
                    labels=list(labels),
                    average="macro",
                    zero_division=0,
                )
            )
            for label in labels:
                np.testing.assert_allclose(
                    predictions[f"{task}_prob_{label}"],
                    [item[task][label] for item in expected],
                    atol=1e-12,
                )
            counts = pd.read_csv(root / f"{task}_confusion_counts.csv", index_col=0)
            assert counts.to_numpy().sum() == len(test)
            assert list(counts.columns) == list(labels)
            assert len(pd.read_csv(root / f"{task}_report.csv")) == len(labels)
            with Image.open(root / f"{task}_confusion.png") as image:
                assert image.format == "PNG" and image.width >= 1000
        assert len(pd.read_csv(root / "errors.csv")) == int((~predictions["joint_correct"]).sum())
        assert (root / "business_formal_errors.csv").is_file()
        assert (root / "error_analysis.md").is_file()
        assert sha256(path / "best.pt") == before[path]
    assert (output / "recurring_errors.csv").is_file()
    assert (output / "business_formal_comparison.csv").is_file()


@pytest.mark.parametrize("problem", ["weights", "labels", "tokenizer", "manifest", "dataset"])
def test_incompatible_inputs_never_publish_reports(saved_models, tmp_path, problem):
    data, paths = saved_models
    copied = tmp_path / "bundle"
    shutil.copytree(paths[0], copied)
    data_copy = tmp_path / "data"
    shutil.copytree(data, data_copy)
    target = {
        "weights": "best.pt",
        "labels": "labels.json",
        "tokenizer": "vocabulary.json",
        "manifest": "data_manifest.json",
    }.get(problem)
    if target:
        with (copied / target).open("ab") as stream:
            stream.write(b"modified")
    else:
        with (data_copy / "test.csv").open("ab") as stream:
            stream.write(b"modified")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="mismatch"):
        evaluate_saved_models([copied], output, data_dir=data_copy, require_all=False)
    assert not output.exists()


def test_a_different_but_valid_dataset_manifest_is_rejected(saved_models, tmp_path):
    data, paths = saved_models
    copied = tmp_path / "data"
    shutil.copytree(data, copied)
    manifest = read_json(copied / "manifest.json")
    manifest["source_sha256"] = "b" * 64
    write_json(copied / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Dataset manifest mismatch"):
        evaluate_saved_models(paths, tmp_path / "reports", data_dir=copied)
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("problem", ["label_order", "vocabulary", "head_size"])
def test_semantic_compatibility_is_checked_even_when_checksums_match(
    saved_models, tmp_path, problem
):
    data, paths = saved_models
    root = tmp_path / "bundle"
    shutil.copytree(paths[0], root)
    filename = {
        "label_order": "labels.json",
        "vocabulary": "vocabulary.json",
        "head_size": "model_config.json",
    }[problem]
    value = read_json(root / filename)
    if problem == "label_order":
        value["styles"] = list(reversed(value["styles"]))
    elif problem == "vocabulary":
        value["tokens"][-1] = "changed-token"
    else:
        value["model_kwargs"]["num_styles"] = 100
    write_json(root / filename, value)
    manifest = read_json(root / "bundle.json")
    manifest["files"][filename] = sha256(root / filename)
    write_json(root / "bundle.json", manifest)
    with pytest.raises(ValueError):
        evaluate_saved_models([root], tmp_path / "reports", data_dir=data, require_all=False)
    assert not (tmp_path / "reports").exists()


def test_export_failure_does_not_leave_a_partially_published_directory(
    saved_models, tmp_path, monkeypatch
):
    import app.evaluation as evaluation

    data, paths = saved_models
    output = tmp_path / "report"

    def broken_export(result, directory, labels):
        directory.mkdir()
        (directory / "partial.txt").write_text("partial")
        raise OSError("Report write failed")

    monkeypatch.setattr(evaluation, "export_model_reports", broken_export)
    with pytest.raises(OSError, match="Report write failed"):
        evaluate_saved_models(paths[:1], output, data_dir=data, require_all=False)
    assert not output.exists()
    assert not list(tmp_path.glob(".report-*"))


def test_output_is_protected_and_all_architectures_are_required(saved_models, tmp_path):
    data, paths = saved_models
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate_saved_models(paths, output, data_dir=data)
    assert (output / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="Missing saved architectures"):
        evaluate_saved_models(paths[:1], tmp_path / "partial", data_dir=data)
    assert not (tmp_path / "partial").exists()


def test_recurring_errors_count_distinct_architectures_and_direction_rates():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e"],
            "text": ["text"] * 5,
            "true_style": ["business", "business", "formal", "formal", "formal"],
            "predicted_style": ["formal", "business", "business", "formal", "academic"],
            "true_tone": ["neutral"] * 5,
            "predicted_tone": ["neutral"] * 5,
            "style_correct": [False, True, False, True, False],
            "tone_correct": [True] * 5,
            "joint_correct": [False, True, False, True, False],
        }
    )
    rates = business_formal_stats(frame)
    assert rates[0]["rate"] == pytest.approx(1 / 2)
    assert rates[1]["rate"] == pytest.approx(1 / 3)
    runs = [
        {"run_id": name, "model_name": "textcnn", "predictions": frame.copy()}
        for name in ("cnn_one", "cnn_two")
    ]
    assert recurring_errors(runs).empty
    runs.append({"run_id": "bilstm", "model_name": "bilstm", "predictions": frame.copy()})
    recurring = recurring_errors(runs)
    assert recurring["sample_id"].tolist() == ["a", "c", "e"]
    assert recurring["joint_error_architectures"].eq("bilstm,textcnn").all()


def test_absent_classes_and_zero_errors_produce_valid_exports(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["one"],
            "text": ["Please send the report."],
            "true_style": ["business"],
            "predicted_style": ["business"],
            "true_tone": ["neutral"],
            "predicted_tone": ["neutral"],
            "style_correct": [True],
            "tone_correct": [True],
            "joint_correct": [True],
            "style_confidence": [1.0],
            "tone_confidence": [1.0],
            "characters": [23],
            "truncated": [False],
            "length_bucket": ["1-80"],
        }
    )
    result = {
        "run_id": "test",
        "model_name": "textcnn",
        "checkpoint": "test/best.pt",
        "metrics": {"joint_accuracy": 1.0},
        "predictions": frame,
        "calibration_bins": {
            f"{task}_{mode}": [] for task in ("style", "tone") for mode in ("raw", "calibrated")
        },
    }
    output = tmp_path / "report"
    export_model_reports(result, output, {"style": STYLES, "tone": TONES})
    assert pd.read_csv(output / "errors.csv").empty
    matrix = pd.read_csv(output / "style_confusion_normalized.csv", index_col=0)
    assert np.isfinite(matrix.to_numpy()).all()
    assert matrix.loc["academic"].sum() == 0
    assert matrix.loc["business", "business"] == 1
    assert "No business/formal swaps" in (output / "error_analysis.md").read_text()
