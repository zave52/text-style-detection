"""Shared training and evaluation for models with style/tone logits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def multitask_loss(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """CrossEntropyLoss expects raw logits and integer labels, without softmax."""
    return F.cross_entropy(outputs["style_logits"], batch["style_labels"]) + F.cross_entropy(
        outputs["tone_logits"], batch["tone_labels"]
    )


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Train when optimizer is supplied; otherwise evaluate without gradients.

    The model must already be on device before constructing its optimizer.
    Metrics are weighted by sample count, including an incomplete final batch.
    """
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    training = optimizer is not None
    model.train(training)
    count = style_correct = tone_correct = 0
    loss_sum = 0.0

    with torch.set_grad_enabled(training):
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=batch["input_ids"], lengths=batch["lengths"])
            loss = multitask_loss(outputs, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite multitask loss.")
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizer.step()

            size = batch["input_ids"].shape[0]
            count += size
            loss_sum += loss.detach().item() * size
            style_correct += (
                (outputs["style_logits"].argmax(dim=1) == batch["style_labels"]).sum().item()
            )
            tone_correct += (
                (outputs["tone_logits"].argmax(dim=1) == batch["tone_labels"]).sum().item()
            )

    if count == 0:
        raise ValueError("Cannot run an epoch on an empty loader.")
    return {
        "loss": loss_sum / count,
        "style_accuracy": style_correct / count,
        "tone_accuracy": tone_correct / count,
    }