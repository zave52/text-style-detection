"""Evaluate saved neural bundles on one verified held-out test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import ARTIFACTS_DIR, PREPARED_DATA_DIR
from app.evaluation import evaluate_saved_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path)
    parser.add_argument("--experiments-root", type=Path, default=ARTIFACTS_DIR / "experiments")
    parser.add_argument("--data-dir", type=Path, default=PREPARED_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--require-all", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    checkpoints = args.checkpoints
    if checkpoints is None:
        checkpoints = sorted(args.experiments_root.rglob("best.pt"))
    if not checkpoints:
        parser.error(
            "No saved best.pt checkpoints found. Supply --checkpoints or --experiments-root."
        )
    evaluate_saved_models(
        checkpoints,
        args.output,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
        bins=args.ece_bins,
        require_all=args.require_all,
    )


if __name__ == "__main__":
    main()
