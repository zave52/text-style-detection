# app/experiments.py

from __future__ import annotations

import inspect
import json
import math
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from matplotlib.figure import Figure

from app.bundles import save_calibrated_bundle
from app.calibration import fit_temperatures
from app.config import SEED, STYLES, TONES, resolve_device, set_seed
from app.data import load_splits
from app.distilbert import PRETRAINED_NAME
from app.models import build_model
from app.tokenization import TransformerTokenizer, Vocabulary, make_loader
from app.training import run_epoch


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    run_dir: str
    epochs: int = 20
    batch_size: int = 32
    max_length: int = 128
    learning_rate: float | None = None
    head_learning_rate: float | None = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    style_weight: float = 1.0
    tone_weight: float = 1.0
    patience: int = 5
    min_delta: float = 0.0
    lr_patience: int = 1
    lr_factor: float = 0.5
    seed: int = SEED
    device: str | None = None
    evaluate_test: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_name not in {"textcnn", "bilstm", "distilbert"}:
            raise ValueError("model_name must be textcnn, bilstm, or distilbert.")
        if not self.run_dir or not isinstance(self.model_kwargs, dict):
            raise ValueError("Provide run_dir and a dictionary of model_kwargs.")
        for value in (self.epochs, self.batch_size, self.max_length, self.patience):
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    "Epoch, batch, length, and patience values must be positive integers."
                )
        if not isinstance(self.lr_patience, int) or self.lr_patience < 0:
            raise ValueError("lr_patience must be a non-negative integer.")
        for value in (self.max_grad_norm, self.style_weight, self.tone_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Gradient norm and task weights must be finite and positive.")
        for value in (self.learning_rate, self.head_learning_rate):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Learning rates must be finite and positive.")
        if any(not math.isfinite(x) or x < 0 for x in (self.weight_decay, self.min_delta)):
            raise ValueError("weight_decay and min_delta must be finite and non-negative.")
        if not math.isfinite(self.lr_factor) or not 0 < self.lr_factor < 1:
            raise ValueError("lr_factor must be between zero and one.")
        if {
            "vocab_size",
            "pad_id",
            "num_styles",
            "num_tones",
            "encoder_config",
        } & self.model_kwargs.keys():
            raise ValueError("Vocabulary, labels, and encoder_config are managed by the pipeline.")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best: float = -math.inf
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if not math.isfinite(score):
            raise FloatingPointError("Validation score is not finite.")
        if score > self.best + self.min_delta:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def plot_history(history: list[dict], destination: Path) -> None:
    figure = Figure(figsize=(14, 4))
    axes = figure.subplots(1, 3)
    epochs = [row["epoch"] for row in history]

    for split in ("train", "validation"):
        axes[0].plot(
            epochs,
            [row[split]["loss"] for row in history],
            label=split,
        )
        axes[1].plot(
            epochs,
            [row[split]["mean_macro_f1"] for row in history],
            label=split,
        )

    for index in range(len(history[0]["learning_rates"])):
        axes[2].plot(
            epochs,
            [row["learning_rates"][index] for row in history],
            label=f"group {index + 1}",
        )

    for axis, title in zip(
        axes,
        ("Combined loss", "Mean macro F1", "Learning rate"),
        strict=True,
    ):
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[1].set_ylim(0, 1.02)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    figure.clear()


def build_optimizer(model, config: TrainConfig) -> torch.optim.AdamW:
    lr = config.learning_rate or (
        2e-5 if config.model_name == "distilbert" else 1e-3
    )

    if config.model_name == "distilbert":
        encoder = list(model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder}
        heads = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in encoder_ids
        ]
        groups = [
            {"params": encoder, "lr": lr},
            {"params": heads, "lr": config.head_learning_rate or 1e-4},
        ]
    else:
        groups = [{"params": model.parameters(), "lr": lr}]

    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def save_best(model, tokenizer, kwargs, config, directory, epoch, validation):
    metadata = {
        "epoch": epoch,
        "validation": validation,
        "training_config": asdict(config),
        "selection_metric": "validation.mean_macro_f1",
    }

    if tokenizer.kind == "transformer":
        model.save_bundle(directory, tokenizer, metadata=metadata)
    else:
        checkpoint = {
            "model_name": config.model_name,
            "model_kwargs": kwargs,
            "state_dict": model.state_dict(),
            "styles": list(STYLES),
            "tones": list(TONES),
            "vocab_tokens": tokenizer.tokens,
            "max_length": tokenizer.max_length,
            "metadata": metadata,
        }
        temporary = directory / "best.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(directory / "best.pt")


def fit(
    model,
    tokenizer,
    kwargs,
    train_loader,
    val_loader,
    config,
    directory,
    device,
):
    optimizer = build_optimizer(model, config)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_factor,
        patience=config.lr_patience,
        threshold=config.min_delta,
        threshold_mode="abs",
    )

    write_json(
        directory / "optimizer.json",
        {
            "name": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "groups": [
                {
                    key: group[key]
                    for key in ("lr", "weight_decay", "betas", "eps")
                }
                for group in optimizer.param_groups
            ],
        },
    )

    stopper = EarlyStopping(config.patience, config.min_delta)
    history, best_score, best_epoch = [], -math.inf, 0

    options = {
        "device": device,
        "max_grad_norm": config.max_grad_norm,
        "style_weight": config.style_weight,
        "tone_weight": config.tone_weight,
    }

    for epoch in range(1, config.epochs + 1):
        rates = [group["lr"] for group in optimizer.param_groups]

        train = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            **options,
        )
        validation = run_epoch(model, val_loader, **options)
        score = validation["mean_macro_f1"]

        if not math.isfinite(score):
            raise FloatingPointError("Validation mean macro F1 is not finite.")

        if score > best_score:
            best_score, best_epoch = score, epoch
            save_best(
                model,
                tokenizer,
                kwargs,
                config,
                directory,
                epoch,
                validation,
            )

        stop = stopper.update(score)
        scheduler.step(score)

        history.append(
            {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "learning_rates": rates,
                "next_learning_rates": [
                    group["lr"] for group in optimizer.param_groups
                ],
            }
        )
        write_json(directory / "history.json", history)

        print(
            f"Epoch {epoch:02d} | loss={train['loss']:.4f} | "
            f"val_loss={validation['loss']:.4f} | "
            f"val_mean_macro_f1={score:.4f}",
            flush=True,
        )

        if stop:
            break

    checkpoint = torch.load(
        directory / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return history, {
        "best_epoch": best_epoch,
        "best_validation_mean_macro_f1": best_score,
        "epochs_completed": len(history),
        "early_stopped": stop,
    }


def run_experiment(config: TrainConfig) -> Path:
    directory = Path(config.run_dir).resolve()

    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Experiment already exists: {directory}. Choose a new --run-dir."
        ) from error

    try:
        write_json(directory / "status.json", {"status": "running"})
        write_json(
            directory / "training_config.json",
            {
                "settings": asdict(config),
                "selection_metric": "validation.mean_macro_f1",
                "started_at": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
            },
        )

        set_seed(config.seed)
        device = resolve_device(config.device)
        frames, manifest = load_splits()

        kwargs = dict(config.model_kwargs)
        kwargs.update(num_styles=len(STYLES), num_tones=len(TONES))

        if config.model_name == "distilbert":
            source = kwargs.get("pretrained_name", PRETRAINED_NAME)
            tokenizer = TransformerTokenizer(
                source,
                max_length=config.max_length,
                local_only=kwargs.get("local_only", False),
            )
        else:
            tokenizer = Vocabulary.fit(
                frames["train"]["text"],
                max_length=config.max_length,
            )
            kwargs.update(
                vocab_size=len(tokenizer),
                pad_id=tokenizer.pad_id,
            )
            tokenizer.save(directory)

        model = build_model(config.model_name, **kwargs).to(device)

        bound = inspect.signature(type(model)).bind(**kwargs)
        bound.apply_defaults()
        kwargs = dict(bound.arguments)

        if tokenizer.kind == "transformer":
            if config.max_length > model.encoder.config.max_position_embeddings:
                raise ValueError(
                    "Sequence limit exceeds encoder positional embeddings."
                )

        train_loader = make_loader(
            frames["train"],
            tokenizer,
            config.batch_size,
            shuffle=True,
            seed=config.seed,
        )
        val_loader = make_loader(
            frames["validation"],
            tokenizer,
            config.batch_size,
        )

        write_json(directory / "data_manifest.json", manifest)
        write_json(directory / "model_kwargs.json", kwargs)

        # fit restores the checkpoint selected by validation mean macro F1.
        history, summary = fit(
            model,
            tokenizer,
            kwargs,
            train_loader,
            val_loader,
            config,
            directory,
            device,
        )

        # Fit both temperatures using validation predictions only.
        calibration = fit_temperatures(
            model,
            frames["validation"],
            tokenizer,
            batch_size=config.batch_size,
            device=device,
        )
        summary["calibration"] = calibration

        if config.evaluate_test:
            test_loader = make_loader(
                frames["test"],
                tokenizer,
                config.batch_size,
            )
            test = run_epoch(
                model,
                test_loader,
                device=device,
                style_weight=config.style_weight,
                tone_weight=config.tone_weight,
                max_grad_norm=config.max_grad_norm,
            )
            write_json(directory / "test_metrics.json", test)

        write_json(directory / "summary.json", summary)

        # Persist calibration in best.pt and checksum all required bundle files.
        save_calibrated_bundle(directory, calibration)

        write_json(directory / "status.json", {"status": "completed"})

        try:
            plot_history(history, directory / "learning_curves.png")
        except Exception as error:
            print(
                f"Warning: learning curves were not generated: {error}",
                flush=True,
            )

        print(
            f"Best epoch: {summary['best_epoch']} | saved to: {directory}",
            flush=True,
        )
        return directory

    except BaseException as error:
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"

        write_json(
            directory / "status.json",
            {
                "status": state,
                "error_type": type(error).__name__,
            },
        )
        raise
