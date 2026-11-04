"""CLI script to compile, audit, and split the corpus into train/validation/test sets."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import (
    CORPUS_AUDIT_JSON,
    PREPARED_DATA_DIR,
    RAW_DATASET_CSV,
    RAW_DATASET_DIR,
    SEED,
)
from app.data import build_and_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_DATASET_DIR,
        help="Path to raw dataset directory with style/tone text files",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=RAW_DATASET_CSV,
        help="Path to the canonical raw CSV",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=PREPARED_DATA_DIR,
        help="Directory to save clean dataset, splits, and manifest",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=CORPUS_AUDIT_JSON.parent,
        help="Directory to save audit and split reports",
    )
    parser.add_argument(
        "--recompile",
        action="store_true",
        help="Force recompilation of raw CSV from dataset/ text files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for group stratified splitting",
    )
    args = parser.parse_args()

    frames, manifest = build_and_audit(
        source_dir=args.source_dir,
        raw_output=args.raw_output,
        prepared_dir=args.prepared_dir,
        reports_dir=args.reports_dir,
        recompile_from_files=args.recompile,
    )

    splits = manifest.get("splits", {})
    train_n = splits.get("train", {}).get("rows", len(frames.get("train", [])))
    val_n = splits.get("validation", {}).get("rows", len(frames.get("validation", [])))
    test_n = splits.get("test", {}).get("rows", len(frames.get("test", [])))

    print(
        f"Done! Cleaned {manifest['clean_rows']} samples into reproducible splits:\n"
        f"  Train:                {train_n} rows ({args.prepared_dir / 'train.csv'})\n"
        f"  Validation:           {val_n} rows ({args.prepared_dir / 'validation.csv'})\n"
        f"  Test:                 {test_n} rows ({args.prepared_dir / 'test.csv'})\n"
        f"  Full clean dataset:   {manifest['clean_rows']} rows\n"
        f"  Excluded conflicts:   {manifest['conflicting_rows_removed']} rows\n"
        f"  Manifest:             {args.prepared_dir / 'manifest.json'}\n"
        f"  Audit report (JSON):  {args.reports_dir / 'corpus_audit.json'}\n"
        f"  Audit report (MD):    {args.reports_dir / 'corpus_audit.md'}\n"
        f"  Splits report (MD):   {args.reports_dir / 'data_splits.md'}"
    )


if __name__ == "__main__":
    main()
