"""Independent temperature scaling for the two classification heads."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from app.tokenization import make_loader
from app.training import model_inputs


@torch.no_grad()
def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Minimize validation NLL; it is convex in inverse temperature."""
    logits = logits.detach().to(device="cpu", dtype=torch.float64)
    labels = labels.detach().to(device="cpu")

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("Expected non-empty logits with shape [samples, classes].")
    if labels.shape != (logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("Expected one int64 label per sample.")
    if not torch.isfinite(logits).all():
        raise ValueError("Calibration logits must be finite.")
    if ((labels < 0) | (labels >= logits.shape[1])).any():
        raise ValueError("Calibration labels are outside the class range.")

    logits = logits - logits.max(dim=1, keepdim=True).values
    correct = logits.gather(1, labels[:, None]).squeeze(1)

    def derivative(inverse_temperature: float) -> float:
        expected = (torch.softmax(logits * inverse_temperature, dim=1) * logits).sum(1)
        return (expected - correct).mean().item()

    # Bound T to [0.05, 100] and include T=1 as the no-calibration fallback.
    low, high = 1 / 100.0, 1 / 0.05

    if abs(derivative(1.0)) < 1e-12:
        inverse = 1.0
    elif derivative(low) >= 0:
        inverse = low
    elif derivative(high) <= 0:
        inverse = high
    else:
        for _ in range(64):
            middle = (low + high) / 2
            if derivative(middle) > 0:
                high = middle
            else:
                low = middle
        inverse = (low + high) / 2

    before = F.cross_entropy(logits, labels).item()
    after = F.cross_entropy(logits * inverse, labels).item()

    if after > before:
        inverse, after = 1.0, before

    return {
        "temperature": 1 / inverse,
        "nll_before": before,
        "nll_after": after,
    }


@torch.no_grad()
def fit_temperatures(
    model,
    validation_data,
    tokenizer,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict:
    """Collect frozen, eval-mode predictions on validation data only."""
    logits = {task: [] for task in ("style", "tone")}
    labels = {task: [] for task in logits}

    was_training = model.training
    model.eval()

    try:
        for batch in make_loader(validation_data, tokenizer, batch_size=batch_size):
            inputs = {
                key: value.to(device)
                for key, value in model_inputs(batch).items()
            }
            outputs = model(**inputs)

            for task in logits:
                logits[task].append(outputs[f"{task}_logits"].detach().cpu())
                labels[task].append(batch[f"{task}_labels"].cpu())
    finally:
        model.train(was_training)

    if not logits["style"]:
        raise ValueError("Cannot calibrate on an empty validation split.")

    result = {
        "method": "temperature_scaling",
        "split": "validation",
    }

    for task in logits:
        result[task] = fit_temperature(
            torch.cat(logits[task]),
            torch.cat(labels[task]),
        )

    result["samples"] = sum(len(batch) for batch in labels["style"])
    return result


def temperatures_from(calibration: dict) -> dict[str, float]:
    if (
        calibration.get("method") != "temperature_scaling"
        or calibration.get("split") != "validation"
    ):
        raise ValueError("Expected temperature scaling fitted on validation predictions.")

    temperatures = {
        task: float(calibration[task]["temperature"])
        for task in ("style", "tone")
    }

    if any(
        not math.isfinite(value) or value <= 0
        for value in temperatures.values()
    ):
        raise ValueError("Temperatures must be finite and strictly positive.")

    return temperatures
