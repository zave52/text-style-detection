from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F


def model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in ("input_ids", "lengths", "attention_mask") if key in batch}


def _check_weights(style_weight: float, tone_weight: float) -> None:
    if any(not math.isfinite(w) or w <= 0 for w in (style_weight, tone_weight)):
        raise ValueError("Both task weights must be finite and positive.")


def task_losses(outputs, batch) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        F.cross_entropy(outputs["style_logits"], batch["style_labels"]),
        F.cross_entropy(outputs["tone_logits"], batch["tone_labels"]),
    )


def multitask_loss(outputs, batch, style_weight: float = 1.0, tone_weight: float = 1.0):
    _check_weights(style_weight, tone_weight)
    style_loss, tone_loss = task_losses(outputs, batch)
    return style_weight * style_loss + tone_weight * tone_loss


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
    style_weight: float = 1.0,
    tone_weight: float = 1.0,
) -> dict[str, float]:
    """Compute macro F1 over the entire epoch and every output class."""
    _check_weights(style_weight, tone_weight)
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive.")
    model.train(optimizer is not None)
    count = 0
    sums = {"loss": 0.0, "style_loss": 0.0, "tone_loss": 0.0}
    targets, predictions, classes = {}, {}, {}
    for task in ("style", "tone"):
        targets[task], predictions[task] = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(**model_inputs(batch))
            style_loss, tone_loss = task_losses(outputs, batch)
            loss = style_weight * style_loss + tone_weight * tone_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite multitask loss.")
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
            size = batch["input_ids"].shape[0]
            count += size
            for key, value in zip(sums, (loss, style_loss, tone_loss), strict=True):
                sums[key] += value.detach().item() * size
            for task in targets:
                logits = outputs[f"{task}_logits"]
                classes[task] = logits.shape[1]
                targets[task].extend(batch[f"{task}_labels"].cpu().tolist())
                predictions[task].extend(logits.detach().argmax(1).cpu().tolist())
    if count == 0:
        raise ValueError("Cannot run an epoch on an empty loader.")
    metrics = {key: value / count for key, value in sums.items()}
    for task in targets:
        metrics[f"{task}_accuracy"] = (
            sum(a == b for a, b in zip(targets[task], predictions[task], strict=True)) / count
        )
        metrics[f"{task}_macro_f1"] = float(
            f1_score(
                targets[task],
                predictions[task],
                labels=list(range(classes[task])),
                average="macro",
                zero_division=0,
            )
        )
    metrics["mean_macro_f1"] = (metrics["style_macro_f1"] + metrics["tone_macro_f1"]) / 2
    return metrics
