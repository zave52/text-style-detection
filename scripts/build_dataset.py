"""CLI script to compile and audit the raw dataset into structured data."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import (
    CORPUS_AUDIT_JSON,
    PREPARED_DATA_DIR,
    RAW_DATASET_CSV,
    RAW_DATASET_DIR,
)
from app.data import build_and_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=RAW_DATASET_DIR,
        help="Path to raw dataset directory with style/tone text files",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=RAW_DATASET_CSV,
        help="Path to save the compiled raw CSV",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=PREPARED_DATA_DIR,
        help="Path to save the clean dataset and excluded conflicts",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=CORPUS_AUDIT_JSON.parent,
        help="Directory to save audit reports (JSON and Markdown)",
    )
    args = parser.parse_args()

    clean_df, audit = build_and_audit(
        source_dir=args.source,
        raw_output=args.raw_output,
        prepared_dir=args.prepared_dir,
        reports_dir=args.reports_dir,
    )

    print(
        f"Done! Compiled {audit['source_rows']} raw samples -> {audit['clean_rows']} clean rows.\n"
        f"  Raw dataset:          {args.raw_output}\n"
        f"  Clean dataset:        {args.prepared_dir / 'dataset.csv'}\n"
        f"  Excluded conflicts:   {args.prepared_dir / 'excluded_conflicts.csv'}\n"
        f"  Audit report (JSON):  {args.reports_dir / 'corpus_audit.json'}\n"
        f"  Audit report (MD):    {args.reports_dir / 'corpus_audit.md'}"
    )


if __name__ == "__main__":
    main()
