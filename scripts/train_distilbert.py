"""Fine-tune the English cased DistilBERT encoder and both classification heads."""

from __future__ import annotations

import argparse
from itertools import chain

import torch

from app.config import ARTIFACTS_DIR, SEED, resolve_device, set_seed
from app.data import load_splits
from app.distilbert import PRETRAINED_NAME
from app.models import build_model
from app.tokenization import TransformerTokenizer, make_loader
from app.training import run_epoch


def main(epochs: int = 5, patience: int = 2, max_length: int = 128, batch_size: int = 8) -> None:
    if min(epochs, patience, batch_size) < 1:
        raise ValueError("epochs, patience, and batch_size must be positive.")
    set_seed(SEED)
    device = resolve_device()
    frames, manifest = load_splits()
    tokenizer = TransformerTokenizer(PRETRAINED_NAME, max_length=max_length)
    model = build_model("distilbert").to(device)
    if max_length > model.encoder.config.max_position_embeddings:
        raise ValueError("max_length exceeds the encoder's position limit.")
    train_loader = make_loader(frames["train"], tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(frames["validation"], tokenizer, batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 2e-5},
            {
                "params": chain(
                    model.pre_classifier.parameters(),
                    model.style_head.parameters(),
                    model.tone_head.parameters(),
                ),
                "lr": 1e-4,
            },
        ],
        weight_decay=0.01,
    )
    output_dir = ARTIFACTS_DIR / "distilbert"
    checkpoint_path = output_dir / "best.pt"
    best_loss, stale_epochs = float("inf"), 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device=device)
        print(f"Epoch {epoch:02d} | train={train_metrics} | validation={val_metrics}")
        if val_metrics["loss"] < best_loss:
            best_loss, stale_epochs = val_metrics["loss"], 0
            model.save_bundle(
                output_dir,
                tokenizer,
                metadata={
                    "seed": SEED,
                    "epoch": epoch,
                    "validation": val_metrics,
                    "data_manifest": manifest,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_loader = make_loader(frames["test"], tokenizer, batch_size=batch_size)
    print("Test:", run_epoch(model, test_loader, device=device))
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(args.epochs, args.patience, args.max_length, args.batch_size)