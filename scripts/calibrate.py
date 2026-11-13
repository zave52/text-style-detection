"""Calibrate an existing shared-pipeline experiment without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.bundles import (
    bundle_directory,
    read_json,
    recover_incomplete_bundle,
    save_calibrated_bundle,
)
from app.calibration import fit_temperatures
from app.config import MAX_BATCH_SIZE, STYLES, TONES
from app.data import load_splits
from app.prediction import NeuralPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recover incomplete calibration; completed bundles remain protected.",
    )
    args = parser.parse_args()

    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}.")

    root = bundle_directory(args.checkpoint)

    if (root / "bundle.json").exists():
        raise FileExistsError("This experiment is already calibrated and sealed.")

    saved_manifest = read_json(root / "data_manifest.json")
    frames, current_manifest = load_splits()

    if saved_manifest != current_manifest:
        raise ValueError("Prepared dataset differs from the experiment's saved manifest.")

    if args.force:
        recover_incomplete_bundle(root)
    elif (root / "calibration.json").exists():
        raise ValueError("Incomplete calibration found. Re-run this command with --force.")

    predictor = NeuralPredictor(root, device=args.device)

    if predictor.styles != STYLES or predictor.tones != TONES:
        raise ValueError("Validation label order does not match the checkpoint.")

    calibration = fit_temperatures(
        predictor.model,
        frames["validation"],
        predictor.tokenizer,
        batch_size=args.batch_size,
        device=predictor.device,
    )

    save_calibrated_bundle(root, calibration)
    print(json.dumps(calibration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
