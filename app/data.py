"""Corpus compilation and audit pipeline.

Compiles raw text files from ``dataset/<style>/<tone>/*.txt`` into a structured
dataset, validates required columns and allowed labels, normalizes legacy
spelling variants, detects duplicate texts and conflicting annotations, and
produces an audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import (
    CORPUS_AUDIT_JSON,
    LABEL_ALIASES,
    PREPARED_DATA_DIR,
    RAW_DATASET_CSV,
    RAW_DATASET_DIR,
    STYLES,
    TONES,
)

REQUIRED_COLUMNS: set[str] = {"text", "style", "tone"}


def text_key(text: str) -> str:
    """Return a whitespace-collapsed, case-folded key for text comparison."""
    return " ".join(text.split()).casefold()


def fingerprint(path_or_bytes: Path | bytes | str) -> str:
    """Compute the SHA-256 hex digest of a file, bytes, or string."""
    if isinstance(path_or_bytes, Path):
        return hashlib.sha256(path_or_bytes.read_bytes()).hexdigest()
    if isinstance(path_or_bytes, str):
        return hashlib.sha256(path_or_bytes.encode("utf-8")).hexdigest()
    return hashlib.sha256(path_or_bytes).hexdigest()


def compile_raw_corpus(source_dir: Path = RAW_DATASET_DIR) -> pd.DataFrame:
    """Compile raw .txt files from ``source_dir/<style>/<tone>/*.txt``.

    Iterates over paths in deterministic sorted order. Empty text files are skipped.
    Returns a DataFrame with columns ``['text', 'style', 'tone']``.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source corpus directory not found: {source_dir}")

    rows: list[dict[str, str]] = []
    for file_path in sorted(source_dir.glob("*/*/*.txt")):
        raw_text = file_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            continue
        rows.append(
            {
                "text": raw_text,
                "style": file_path.parent.parent.name,
                "tone": file_path.parent.name,
            }
        )

    if not rows:
        raise ValueError(f"No non-empty text files found in {source_dir}")

    return pd.DataFrame(rows)


def read_corpus(source: Path | pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, clean, and audit the source corpus.

    Parameters
    ----------
    source:
        Path to a CSV file or an already-loaded pandas DataFrame with at least
        ``text``, ``style``, and ``tone`` columns.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]:
        The cleaned DataFrame (columns: ``text``, ``style``, ``tone``, ``sample_id``)
        and an audit metadata dictionary detailing removals, alias fixes, and
        checksums.
    """
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"Source dataset file not found: {source}")
        frame = pd.read_csv(source, keep_default_na=False)
        src_sha256 = fingerprint(source)
    elif isinstance(source, pd.DataFrame):
        frame = source.copy()
        src_sha256 = fingerprint(frame.to_csv(index=False).encode("utf-8"))
    else:
        raise TypeError(f"source must be Path or pd.DataFrame, got {type(source).__name__}")

    if not REQUIRED_COLUMNS.issubset(frame.columns):
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        raise ValueError(
            f"Corpus is missing required columns: {missing}. Found: {list(frame.columns)}"
        )

    frame = frame[["text", "style", "tone"]].copy()
    raw_rows = len(frame)

    # 1. Clean strings and normalize whitespace
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["style"] = frame["style"].astype(str).str.strip().str.lower()
    frame["tone"] = frame["tone"].astype(str).str.strip().str.lower()

    # 2. Fix known label aliases (e.g. literaly -> literary)
    aliases_fixed = 0
    for alias, canonical in LABEL_ALIASES.items():
        count = int((frame["style"] == alias).sum() + (frame["tone"] == alias).sum())
        aliases_fixed += count
        frame["style"] = frame["style"].replace({alias: canonical})
        frame["tone"] = frame["tone"].replace({alias: canonical})

    # 3. Filter empty records
    empty_mask = frame["text"].str.len().eq(0)
    empty_count = int(empty_mask.sum())
    frame = frame.loc[~empty_mask].copy()

    # 4. Validate allowed classes and ensure all project classes are represented
    for target, allowed in [("style", STYLES), ("tone", TONES)]:
        invalid = set(frame[target]) - set(allowed)
        if invalid:
            raise ValueError(
                f"Unknown {target} labels: {sorted(invalid)}. Allowed: {list(allowed)}"
            )
        if set(frame[target]) != set(allowed):
            missing_classes = sorted(set(allowed) - set(frame[target]))
            raise ValueError(f"Missing required {target} classes: {missing_classes}")

    # 5. Identify conflicting annotations and exact duplicates
    frame["_key"] = frame["text"].map(text_key)

    conflicts = frame.groupby("_key")[["style", "tone"]].nunique().max(axis=1)
    conflict_keys = set(conflicts[conflicts > 1].index)
    excluded_conflicts_df = frame.loc[
        frame["_key"].isin(conflict_keys), ["text", "style", "tone"]
    ].copy()

    # A conflicting annotation has no defensible automatic winner: exclude every copy.
    frame = frame.loc[~frame["_key"].isin(conflict_keys)].copy()

    # Deduplicate exact normalized texts with agreed labels
    before_dedup = len(frame)
    frame = frame.drop_duplicates("_key").reset_index(drop=True)
    duplicates_removed = before_dedup - len(frame)

    # 6. Assign reproducible sample_id
    frame["sample_id"] = frame["_key"].map(
        lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest()[:16]
    )

    clean_frame = frame.drop(columns="_key")

    audit: dict[str, Any] = {
        "source_sha256": src_sha256,
        "source_rows": raw_rows,
        "empty_removed": empty_count,
        "duplicates_removed": duplicates_removed,
        "conflicting_groups_removed": len(conflict_keys),
        "conflicting_rows_removed": len(excluded_conflicts_df),
        "excluded_conflicts": excluded_conflicts_df.to_dict(orient="records"),
        "label_aliases_fixed": aliases_fixed,
        "clean_rows": len(clean_frame),
        "class_distribution": {
            "style": {k: int(v) for k, v in clean_frame["style"].value_counts().items()},
            "tone": {k: int(v) for k, v in clean_frame["tone"].value_counts().items()},
        },
    }

    return clean_frame, audit


def generate_audit_markdown(audit: dict[str, Any]) -> str:
    """Generate a readable Markdown report from the audit summary."""
    lines: list[str] = [
        "# Corpus Audit Report",
        "",
        "## Summary",
        "",
        "| Metric | Count | Note |",
        "|---|---:|---|",
        f"| Source records | {audit['source_rows']} | Initial input rows |",
        f"| Empty texts removed | {audit['empty_removed']} | Length = 0 after whitespace |",
        f"| Duplicate rows removed | {audit['duplicates_removed']} | Identical texts and labels |",
        (
            f"| Conflicting groups removed | {audit['conflicting_groups_removed']} | "
            "Identical texts with conflicting labels |"
        ),
        (
            f"| Conflicting rows excluded | {audit['conflicting_rows_removed']} | "
            "Preserved for manual inspection |"
        ),
        f"| Label aliases normalized | {audit['label_aliases_fixed']} | `literaly` → `literary` |",
        (
            f"| **Clean verified records** | **{audit['clean_rows']}** | "
            "Ready for splitting and modeling |"
        ),
        "",
    ]

    if audit.get("excluded_conflicts"):
        lines.extend(
            [
                "## Excluded Conflicting Annotations",
                "",
                "Identical normalized texts with conflicting annotations were identified.",
                "Instead of picking an arbitrary label, all conflicting copies are excluded:",
                "",
                "| Text | Style | Tone |",
                "|---|---|---|",
            ]
        )
        for row in audit["excluded_conflicts"]:
            lines.append(f"| `{row['text']}` | `{row['style']}` | `{row['tone']}` |")
        lines.append("")

    if "class_distribution" in audit:
        lines.extend(
            [
                "## Clean Corpus Class Distribution",
                "",
                "### Writing Styles",
                "",
                "| Style | Count |",
                "|---|---:|",
            ]
        )
        for style, count in sorted(audit["class_distribution"]["style"].items()):
            lines.append(f"| `{style}` | {count} |")

        lines.extend(
            [
                "",
                "### Communicative Tones",
                "",
                "| Tone | Count |",
                "|---|---:|",
            ]
        )
        for tone, count in sorted(audit["class_distribution"]["tone"].items()):
            lines.append(f"| `{tone}` | {count} |")
        lines.append("")

    return "\n".join(lines)


def build_and_audit(
    source_dir: Path = RAW_DATASET_DIR,
    raw_output: Path | None = RAW_DATASET_CSV,
    prepared_dir: Path = PREPARED_DATA_DIR,
    reports_dir: Path = CORPUS_AUDIT_JSON.parent,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute the end-to-end corpus compilation and audit.

    1. Compiles raw text files from ``source_dir``.
    2. Saves the compiled raw dataset to ``raw_output`` (if provided).
    3. Runs ``read_corpus`` audit.
    4. Writes ``dataset.csv`` and ``excluded_conflicts.csv`` to ``prepared_dir``.
    5. Writes audit reports (JSON and Markdown) to ``reports_dir``.
    """
    raw_df = compile_raw_corpus(source_dir)

    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_output, index=False, encoding="utf-8")

    clean_df, audit = read_corpus(raw_df)

    prepared_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(prepared_dir / "dataset.csv", index=False, encoding="utf-8")

    if audit["excluded_conflicts"]:
        pd.DataFrame(audit["excluded_conflicts"]).to_csv(
            prepared_dir / "excluded_conflicts.csv", index=False, encoding="utf-8"
        )

    audit_json = json.dumps(audit, indent=2)
    (prepared_dir / "corpus_audit.json").write_text(audit_json, encoding="utf-8")

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "corpus_audit.json"
    md_path = reports_dir / "corpus_audit.md"

    json_path.write_text(audit_json, encoding="utf-8")
    md_path.write_text(generate_audit_markdown(audit), encoding="utf-8")

    return clean_df, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=RAW_DATASET_DIR, help="Source dataset directory"
    )
    parser.add_argument(
        "--raw-output", type=Path, default=RAW_DATASET_CSV, help="Path for raw compiled CSV"
    )
    parser.add_argument(
        "--prepared-dir", type=Path, default=PREPARED_DATA_DIR, help="Directory for clean CSVs"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=CORPUS_AUDIT_JSON.parent,
        help="Directory for audit reports",
    )
    args = parser.parse_args()

    clean_df, audit = build_and_audit(
        source_dir=args.source,
        raw_output=args.raw_output,
        prepared_dir=args.prepared_dir,
        reports_dir=args.reports_dir,
    )
    print(
        f"Audited {audit['source_rows']} source rows -> "
        f"{audit['clean_rows']} clean rows saved to {args.prepared_dir}"
    )
    print(f"Audit report saved to {args.reports_dir / 'corpus_audit.json'}")


if __name__ == "__main__":
    main()
