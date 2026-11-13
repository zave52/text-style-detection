"""Evaluate compatible neural bundles on one immutable held-out test split."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F

from app.bundles import bundle_directory, read_json, sha256, verify_bundle, write_json
from app.config import MAX_BATCH_SIZE, PREPARED_DATA_DIR, STYLES, TONES
from app.data import load_splits
from app.evaluation_reports import export_comparison, export_model_reports
from app.prediction import NeuralPredictor
from app.tokenization import make_loader
from app.training import model_inputs

ARCHITECTURES = {"textcnn", "bilstm", "distilbert"}
LABELS = {"style": STYLES, "tone": TONES}


@torch.no_grad()
def probability_metrics(logits, targets, temperature: float, bins: int) -> tuple[dict, list]:
    """NLL, unscaled multiclass Brier score, and top-label equal-width ECE."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.long)
    if bins < 1 or logits.ndim != 2 or len(targets) == 0:
        raise ValueError("Probability metrics require non-empty logits and positive bins.")
    if not np.isfinite(temperature) or temperature <= 0 or not torch.isfinite(logits).all():
        raise ValueError("Logits and positive temperature must be finite.")
    scaled = logits / temperature
    probabilities = scaled.softmax(1).numpy()
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == targets.numpy()
    assignment = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        average_confidence = float(confidence[mask].mean()) if count else None
        gap = abs(accuracy - average_confidence) if count else 0.0
        ece += count / len(targets) * gap
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "samples": count,
                "accuracy": accuracy,
                "confidence": average_confidence,
            }
        )
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).numpy()
    return {
        "nll": F.cross_entropy(scaled, targets).item(),
        "brier": float(np.square(probabilities - one_hot).sum(1).mean()),
        "ece": float(ece),
    }, rows


def validate_checkpoint(path: Path, dataset_manifest: dict) -> tuple[Path, dict]:
    root = bundle_directory(path)
    if not (root / "bundle.json").is_file():
        raise ValueError(f"Evaluation requires a calibrated shared-pipeline bundle: {root}")
    bundle = verify_bundle(root)
    if read_json(root / "data_manifest.json") != dataset_manifest:
        raise ValueError(f"Dataset manifest mismatch for {root}.")
    if read_json(root / "labels.json") != {"styles": list(STYLES), "tones": list(TONES)}:
        raise ValueError(f"Saved label order is incompatible with the dataset: {root}.")
    config = read_json(root / "model_config.json")
    if config["model_name"] != bundle["model_name"]:
        raise ValueError(f"Architecture mismatch in {root}.")
    expected_kind = "transformer" if bundle["model_name"] == "distilbert" else "vocabulary"
    if config["tokenizer_kind"] != expected_kind:
        raise ValueError(f"Tokenizer kind mismatch in {root}.")
    for task, labels in LABELS.items():
        if config["model_kwargs"][f"num_{task}s"] != len(labels):
            raise ValueError(f"Classification head size mismatch in {root}.")
    return root, bundle


@torch.inference_mode()
def collect_logits(predictor: NeuralPredictor, frame: pd.DataFrame, batch_size: int) -> dict:
    predictor.model.eval()
    collected = {task: [] for task in LABELS}
    for batch in make_loader(frame, predictor.tokenizer, batch_size=batch_size, shuffle=False):
        inputs = {key: value.to(predictor.device) for key, value in model_inputs(batch).items()}
        outputs = predictor.model(**inputs)
        for task, labels in LABELS.items():
            logits = outputs[f"{task}_logits"]
            if logits.shape != (len(batch["input_ids"]), len(labels)):
                raise ValueError(f"Invalid {task} output dimensions.")
            if not torch.isfinite(logits).all():
                raise ValueError(f"Non-finite {task} logits.")
            collected[task].append(logits.detach().cpu().double())
    return {task: torch.cat(parts).numpy() for task, parts in collected.items()}


def evaluate_model(
    predictor, frame, *, batch_size: int, bins: int
) -> tuple[dict, pd.DataFrame, dict]:
    logits = collect_logits(predictor, frame, batch_size)
    predictions = frame[["sample_id", "text", "style", "tone"]].copy()
    predictions = predictions.rename(columns={"style": "true_style", "tone": "true_tone"})
    inspected = [predictor.tokenizer.inspect(text) for text in frame["text"]]
    for key in ("token_count", "truncated", "unknown_ratio"):
        predictions[key] = [row[key] for row in inspected]
    predictions["characters"] = frame["text"].str.len().to_numpy()
    predictions["length_bucket"] = pd.cut(
        predictions["characters"], bins=[0, 80, 200, np.inf], labels=["1-80", "81-200", "201+"]
    ).astype(str)
    metrics, calibration_bins = {}, {}
    for task, labels in LABELS.items():
        targets = (
            frame[task]
            .map({label: index for index, label in enumerate(labels)})
            .to_numpy(copy=True)
        )
        predicted = logits[task].argmax(1)
        predictions[f"predicted_{task}"] = np.asarray(labels)[predicted]
        predictions[f"{task}_correct"] = predicted == targets
        probabilities = torch.as_tensor(logits[task]).div(predictor.temperatures[task]).softmax(1)
        predictions[f"{task}_confidence"] = probabilities.max(1).values.numpy()
        predictions[f"{task}_nll"] = F.cross_entropy(
            torch.as_tensor(logits[task]) / predictor.temperatures[task],
            torch.as_tensor(targets),
            reduction="none",
        ).numpy()
        for index, label in enumerate(labels):
            predictions[f"{task}_prob_{label}"] = probabilities[:, index].numpy()
        metrics[task] = {
            "accuracy": float(accuracy_score(targets, predicted)),
            "macro_f1": float(
                f1_score(
                    targets, predicted, labels=range(len(labels)), average="macro", zero_division=0
                )
            ),
            "weighted_f1": float(
                f1_score(
                    targets,
                    predicted,
                    labels=range(len(labels)),
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
        for mode, temperature in (("raw", 1.0), ("calibrated", predictor.temperatures[task])):
            scores, rows = probability_metrics(logits[task], targets, temperature, bins)
            metrics[task][mode] = scores
            calibration_bins[f"{task}_{mode}"] = rows
    predictions["joint_correct"] = predictions["style_correct"] & predictions["tone_correct"]
    metrics["joint_accuracy"] = float(predictions["joint_correct"].mean())
    metrics["mean_macro_f1"] = (metrics["style"]["macro_f1"] + metrics["tone"]["macro_f1"]) / 2
    return metrics, predictions, calibration_bins


def evaluate_saved_models(
    checkpoints,
    output: str | Path,
    *,
    data_dir: str | Path = PREPARED_DATA_DIR,
    batch_size: int = 32,
    device: str | None = None,
    bins: int = 15,
    require_all: bool = True,
) -> Path:
    if not 1 <= batch_size <= MAX_BATCH_SIZE or not isinstance(bins, int) or bins < 1:
        raise ValueError(f"Use batch_size=1..{MAX_BATCH_SIZE} and a positive integer bin count.")
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output}")
    paths = [bundle_directory(path) for path in checkpoints]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide at least one checkpoint and no duplicate paths.")
    if output.is_relative_to(data_dir) or any(output.is_relative_to(path) for path in paths):
        raise ValueError("Evaluation output must be outside dataset and checkpoint directories.")
    frames, dataset_manifest = load_splits(data_dir)
    if set(frames) != {"train", "validation", "test"}:
        raise ValueError("The prepared dataset must contain train, validation, and test splits.")
    for split, frame in frames.items():
        if len(frame) != dataset_manifest["splits"][split]["rows"]:
            raise ValueError(f"Manifest row count mismatch: {split}")
    test = frames["test"].reset_index(drop=True).copy()
    if not {"sample_id", "text", "style", "tone"}.issubset(test.columns):
        raise ValueError("Test split is missing required columns.")
    if test["sample_id"].isna().any():
        raise ValueError("Test sample IDs must be present.")
    test["sample_id"] = test["sample_id"].astype(str)
    if test["sample_id"].duplicated().any() or test["sample_id"].str.strip().eq("").any():
        raise ValueError("Test sample IDs must be unique and non-empty.")
    if any(not isinstance(text, str) or not text.strip() for text in test["text"]):
        raise ValueError("Test texts must be non-empty strings.")
    for task, labels in LABELS.items():
        if not test[task].isin(labels).all():
            raise ValueError(f"Unknown test {task} labels.")
    validated = [validate_checkpoint(path, dataset_manifest) for path in paths]
    available = {bundle["model_name"] for _, bundle in validated}
    if require_all and available != ARCHITECTURES:
        raise ValueError(
            f"Missing saved architectures: {', '.join(sorted(ARCHITECTURES - available))}"
        )
    results = []
    for index, (root, bundle) in enumerate(validated, start=1):
        print(f"Evaluating {bundle['model_name']}: {root}", flush=True)
        predictor = NeuralPredictor(root, device=device)
        if predictor.styles != STYLES or predictor.tones != TONES:
            raise ValueError(f"Checkpoint labels do not match the shared dataset: {root}")
        scores, predictions, calibration_bins = evaluate_model(
            predictor, test, batch_size=batch_size, bins=bins
        )
        run_id = f"{index:02d}_{predictor.model_name}"
        results.append(
            {
                "run_id": run_id,
                "model_name": predictor.model_name,
                "checkpoint": str(root / "best.pt"),
                "weights_sha256": bundle["weights_sha256"],
                "bundle_sha256": sha256(root / "bundle.json"),
                "metadata": predictor.metadata,
                "max_length": predictor.tokenizer.max_length,
                "temperatures": predictor.temperatures,
                "metrics": scores,
                "predictions": predictions,
                "calibration_bins": calibration_bins,
            }
        )
        del predictor
    # All bundles, tokenizer/config compatibility, and predictions are checked before publishing.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        write_json(stage / "dataset_manifest.json", dataset_manifest)
        write_json(
            stage / "evaluation_config.json",
            {
                "split": "test",
                "rows": len(test),
                "test_sha256": dataset_manifest["splits"]["test"]["sha256"],
                "sample_order_sha256": hashlib.sha256(
                    json.dumps(test["sample_id"].tolist()).encode()
                ).hexdigest(),
                "batch_size": batch_size,
                "device": device,
                "ece_bins": bins,
                "brier_definition": (
                    "mean over samples of sum over classes of (p - one_hot)^2; range [0,2]"
                ),
                "ece_definition": (
                    "top-label ECE; equal-width bins; confidence=1 belongs to the last bin"
                ),
                "macro_f1_definition": (
                    "unweighted mean over all configured classes; zero_division=0"
                ),
                "created_at": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "architectures": sorted(available),
            },
        )
        for result in results:
            export_model_reports(result, stage / result["run_id"], LABELS)
        export_comparison(results, stage)
        if output.exists():
            raise FileExistsError(f"Evaluation directory already exists: {output}")
        stage.rename(output)
    print(f"Evaluation saved to: {output}", flush=True)
    return output
