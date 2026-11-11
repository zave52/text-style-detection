"""Regression tests for metrics, model selection, scheduling, and experiment isolation."""

import json

import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn

import app.experiments as experiments
from app.config import STYLES, TONES
from app.experiments import TrainConfig, run_experiment
from app.prediction import NeuralPredictor
from app.training import multitask_loss, run_epoch


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def example_frames():
    frame = pd.DataFrame(
        {
            "text": [
                "Hello world please send report now!",
                "hello please send report",
                "please send report now!",
                "Hello world!",
                "send report now please",
            ],
            "style": list(STYLES),
            "tone": list(TONES),
        }
    )
    return {name: frame.copy() for name in ("train", "validation", "test")}


def test_metrics_use_full_epoch_and_weighted_losses():
    labels = torch.tensor([0, 0, 1, 2, 2])
    predicted = torch.tensor([0, 1, 1, 1, 2])
    logits = torch.nn.functional.one_hot(predicted, num_classes=5).float() * 4

    class Lookup(nn.Module):
        def forward(self, input_ids, lengths=None):
            return {"style_logits": logits[input_ids[:, 0]], "tone_logits": logits[input_ids[:, 0]]}

    batches = [
        {
            "input_ids": torch.arange(start, end).unsqueeze(1),
            "style_labels": labels[start:end],
            "tone_labels": labels[start:end],
        }
        for start, end in ((0, 3), (3, 5))
    ]
    metrics = run_epoch(Lookup(), batches, style_weight=2.0, tone_weight=0.5)
    expected = f1_score(labels, predicted, labels=range(5), average="macro", zero_division=0)
    assert metrics["mean_macro_f1"] == pytest.approx(expected)
    assert metrics["style_accuracy"] == pytest.approx(3 / 5)
    ce = torch.nn.functional.cross_entropy(logits, labels).item()
    assert metrics["style_loss"] == pytest.approx(ce)
    assert metrics["loss"] == pytest.approx(2.5 * ce)
    outputs = {"style_logits": logits, "tone_logits": logits}
    batch = {"style_labels": labels, "tone_labels": labels}
    assert multitask_loss(outputs, batch, 2.0, 0.5).item() == pytest.approx(metrics["loss"])


def test_gradient_clipping_bounds_the_optimizer_update():
    class LargeGradient(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(5))

        def forward(self, input_ids, lengths=None):
            logits = (self.weight * 10000).expand(input_ids.shape[0], -1)
            return {"style_logits": logits, "tone_logits": logits}

    model = LargeGradient()
    batch = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "style_labels": torch.zeros(2, dtype=torch.long),
        "tone_labels": torch.zeros(2, dtype=torch.long),
    }
    run_epoch(
        model, [batch], optimizer=torch.optim.SGD(model.parameters(), lr=1), max_grad_norm=0.1
    )
    assert 0 < model.weight.detach().norm().item() <= 0.10001


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_experiment_is_never_touched(tmp_path, monkeypatch, nonempty):
    root = tmp_path / "existing"
    root.mkdir()
    if nonempty:
        (root / "best.pt").write_bytes(b"existing checkpoint")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.setattr(experiments, "load_splits", lambda: pytest.fail("Data must not be read"))
    with pytest.raises(FileExistsError):
        run_experiment(TrainConfig("textcnn", str(root)))
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize("evaluate_test", [False, True])
@pytest.mark.parametrize(
    "scores,delta,best_epoch", [([0.2, 0.6, 0.59, 0.58], 0.0, 2), ([0.5, 0.51, 0.515], 0.1, 3)]
)
def test_selection_early_stopping_scheduler_and_test_isolation(
    tmp_path, monkeypatch, evaluate_test, scores, delta, best_epoch
):
    frames = example_frames()
    trace, captured = [], {}
    for name, frame in frames.items():
        frame.attrs["split"] = name

    def loader(frame, *args, **kwargs):
        if frame.attrs["split"] == "test":
            assert trace.count("validation") == len(scores)
            assert torch.all(captured["model"].style_head.bias == best_epoch)
        return frame.attrs["split"]

    def epoch(model, split, optimizer=None, **kwargs):
        trace.append(split)
        captured["model"] = model
        if split == "train":
            with torch.no_grad():
                model.style_head.bias.fill_(trace.count("train"))
        index = trace.count("validation") - 1
        score = scores[index] if split == "validation" else 0.1
        # Loss and accuracy favour later epochs; only macro F1 may select weights.
        return {
            "loss": 1 / len(trace),
            "mean_macro_f1": score,
            "style_macro_f1": score,
            "tone_macro_f1": score,
            "style_accuracy": 1.0,
        }

    monkeypatch.setattr(experiments, "load_splits", lambda: (frames, {}))
    monkeypatch.setattr(experiments, "make_loader", loader)
    monkeypatch.setattr(experiments, "run_epoch", epoch)
    root = tmp_path / "selection"
    run_experiment(
        TrainConfig(
            "textcnn",
            str(root),
            epochs=10,
            patience=2,
            min_delta=delta,
            lr_patience=0,
            lr_factor=0.5,
            learning_rate=0.01,
            evaluate_test=evaluate_test,
            model_kwargs={"embedding_dim": 4, "num_filters": 2},
        )
    )
    summary = read_json(root / "summary.json")
    assert summary["best_epoch"] == best_epoch and summary["early_stopped"]
    assert summary["epochs_completed"] == len(scores)
    checkpoint = torch.load(root / "best.pt", weights_only=True)
    assert checkpoint["metadata"]["epoch"] == best_epoch
    assert torch.all(checkpoint["state_dict"]["style_head.bias"] == best_epoch)
    history = read_json(root / "history.json")
    assert history[-1]["next_learning_rates"][0] < history[0]["learning_rates"][0]
    assert trace == [item for _ in scores for item in ("train", "validation")] + (
        ["test"] if evaluate_test else []
    )
    assert (root / "test_metrics.json").exists() == evaluate_test


def test_failed_runs_remain_protected(tmp_path, monkeypatch):
    def broken_data():
        raise ValueError("Bad data manifest")

    monkeypatch.setattr(experiments, "load_splits", broken_data)
    config = TrainConfig("bilstm", str(tmp_path / "failed"))
    with pytest.raises(ValueError, match="Bad data"):
        run_experiment(config)
    assert read_json(tmp_path / "failed" / "status.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        run_experiment(config)


def test_recorded_config_is_reusable_and_cli_overrides_it(tmp_path, monkeypatch):
    import scripts.train as cli

    source = tmp_path / "training_config.json"
    source.write_text(
        json.dumps(
            {
                "settings": {
                    "model_name": "textcnn",
                    "run_dir": "old_run",
                    "epochs": 3,
                    "learning_rate": 0.01,
                    "evaluate_test": True,
                }
            }
        ),
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr(cli, "run_experiment", captured.append)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--config",
            str(source),
            "--run-dir",
            "new_run",
            "--epochs",
            "7",
            "--no-evaluate-test",
        ],
    )
    cli.main()
    assert len(captured) == 1
    assert captured[0].run_dir == "new_run" and captured[0].epochs == 7
    assert captured[0].learning_rate == 0.01 and not captured[0].evaluate_test


@pytest.fixture
def local_encoder(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    words = "[PAD] [UNK] [CLS] [SEP] [MASK] Hello hello world please send report now !".split()
    backend = Tokenizer(WordPiece({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    source = tmp_path / "local_encoder"
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
            dim=16,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=32,
        )
    ).save_pretrained(source)
    return source


@pytest.mark.parametrize("name", ["textcnn", "bilstm", "distilbert"])
def test_all_architectures_train_save_plot_and_predict(tmp_path, monkeypatch, local_encoder, name):
    monkeypatch.setattr(experiments, "load_splits", lambda: (example_frames(), {"seed": 42}))
    architecture = {
        "textcnn": {"embedding_dim": 8, "num_filters": 4},
        "bilstm": {"embedding_dim": 8, "hidden_size": 6, "attention_dim": 4},
        "distilbert": {"pretrained_name": str(local_encoder), "local_only": True},
    }[name]
    root = tmp_path / name
    run_experiment(
        TrainConfig(
            name,
            str(root),
            epochs=2,
            batch_size=3,
            max_length=16,
            model_kwargs=architecture,
            evaluate_test=True,
        )
    )
    assert read_json(root / "status.json")["status"] == "completed"
    assert read_json(root / "training_config.json")["settings"]["model_name"] == name
    assert read_json(root / "optimizer.json")["name"] == "AdamW"
    assert len(read_json(root / "history.json")) == 2
    assert "mean_macro_f1" in read_json(root / "test_metrics.json")
    with Image.open(root / "learning_curves.png") as image:
        assert image.format == "PNG" and image.width > 1000
    if name == "distilbert":
        assert read_json(root / "config.json")["model_type"] == "distilbert"
    predictor = NeuralPredictor(root, device="cpu")
    assert set(predictor.predict("Hello world please send report now!")[0]) == {"style", "tone"}
