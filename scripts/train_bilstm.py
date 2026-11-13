from __future__ import annotations

import argparse

import torch

from app.config import ARTIFACTS_DIR, SEED, STYLES, TONES, resolve_device, set_seed
from app.data import load_splits
from app.models import build_model
from app.tokenization import Vocabulary, make_loader
from app.training import run_epoch


def main(epochs: int = 30, patience: int = 5) -> None:
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive.")
    set_seed(SEED)
    device = resolve_device()
    frames, manifest = load_splits()
    vocab = Vocabulary.fit(frames["train"]["text"])
    train_loader = make_loader(frames["train"], vocab, batch_size=32, shuffle=True)
    val_loader = make_loader(frames["validation"], vocab, batch_size=64)
    model_kwargs = {
        "vocab_size": len(vocab),
        "embedding_dim": 128,
        "hidden_size": 128,
        "num_layers": 1,
        "attention_dim": 64,
        "dropout": 0.3,
        "pad_id": vocab.pad_id,
        "num_styles": len(STYLES),
        "num_tones": len(TONES),
    }
    model = build_model("bilstm", **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    output_dir = ARTIFACTS_DIR / "bilstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(output_dir)
    checkpoint_path = output_dir / "best.pt"
    best_loss, stale_epochs = float("inf"), 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device=device)
        print(f"Epoch {epoch:02d} | train={train_metrics} | validation={val_metrics}")
        if val_metrics["loss"] < best_loss:
            best_loss, stale_epochs = val_metrics["loss"], 0
            torch.save(
                {
                    "model_name": "bilstm",
                    "model_kwargs": model_kwargs,
                    "state_dict": model.state_dict(),
                    "styles": list(STYLES),
                    "tones": list(TONES),
                    "vocab_tokens": vocab.tokens,
                    "max_length": vocab.max_length,
                    "seed": SEED,
                    "epoch": epoch,
                    "validation": val_metrics,
                    "data_manifest": manifest,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_loader = make_loader(frames["test"], vocab, batch_size=64)
    print("Test:", run_epoch(model, test_loader, device=device))
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    main(epochs=args.epochs, patience=args.patience)
