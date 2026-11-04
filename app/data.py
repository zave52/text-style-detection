"""Corpus compilation, validation, audit, and reproducible splitting pipeline.

Compiles raw text files from ``dataset/<style>/<tone>/*.txt`` into a structured
dataset, validates required columns and allowed labels, normalizes legacy
spelling variants, detects duplicate texts and conflicting annotations, clusters
near-duplicates into similarity groups, and partitions the corpus into
leakage-free train, validation, and test splits with a verified manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.config import (
    CORPUS_AUDIT_JSON,
    LABEL_ALIASES,
    PREPARED_DATA_DIR,
    RAW_DATASET_CSV,
    RAW_DATASET_DIR,
    SEED,
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


def similarity_groups(texts: list[str], threshold: float = 0.82) -> list[int]:
    """Conservatively group near-copies using word trigrams before splitting.

    Also groups a nearly contained passage when it is at least half as long.
    This serves as an unsupervised data-leakage safeguard.
    """
    parent = list(range(len(texts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grams: list[set[tuple[str, ...]]] = []
    for text in texts:
        words = re.findall(r"\w+", text.casefold())
        grams.append({tuple(words[i : i + 3]) for i in range(len(words) - 2)})

    for i, a in enumerate(grams):
        if len(a) < 8:
            continue
        for j in range(i):
            b = grams[j]
            shorter, longer = min(len(a), len(b)), max(len(a), len(b))
            if shorter < 8 or shorter / longer < 0.5:
                continue
            intersection = len(a & b)
            union = len(a) + len(b) - intersection
            if intersection / union >= threshold or intersection / shorter >= 0.95:
                ra, rb = root(i), root(j)
                parent[max(ra, rb)] = min(ra, rb)

    return [root(i) for i in range(len(texts))]


def verify_splits(frames: dict[str, pd.DataFrame]) -> None:
    """Verify that train, validation, and test splits have zero data leakage."""
    names = list(frames)
    for i, name in enumerate(names):
        if frames[name].empty:
            raise ValueError(f"Empty {name} split")
        for other in names[:i]:
            for column in ["sample_id", "group_id"]:
                if column in frames[name].columns and column in frames[other].columns:
                    overlap = set(frames[name][column]) & set(frames[other][column])
                    if overlap:
                        raise ValueError(
                            f"Data leakage between {name} and {other}: {column} overlap: {overlap}"
                        )
            overlap_texts = set(frames[name].text.map(text_key)) & set(
                frames[other].text.map(text_key)
            )
            if overlap_texts:
                raise ValueError(
                    f"Duplicate text crosses split boundary between {name} and {other}"
                )


def prepare_splits(
    source: Path | pd.DataFrame = RAW_DATASET_CSV,
    destination: Path = PREPARED_DATA_DIR,
    reports_dir: Path = CORPUS_AUDIT_JSON.parent,
    seed: int = SEED,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Partition the audited corpus into reproducible, leakage-free splits.

    Uses nested StratifiedGroupKFold on joint ``style__tone`` strata to keep
    near-duplicate text groups together while maintaining class balance.
    """
    frame, audit = read_corpus(source)
    frame["group_id"] = similarity_groups(frame.text.tolist())

    strata = frame["style"] + "__" + frame["tone"]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(frame, strata, frame.group_id))
    trainval = frame.iloc[trainval_idx].reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_idx, val_idx = next(
        inner.split(trainval, trainval["style"] + "__" + trainval["tone"], trainval.group_id)
    )

    frames: dict[str, pd.DataFrame] = {
        "train": trainval.iloc[train_idx].reset_index(drop=True),
        "validation": trainval.iloc[val_idx].reset_index(drop=True),
        "test": frame.iloc[test_idx].reset_index(drop=True),
    }

    verify_splits(frames)

    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "dataset.csv", index=False)

    if audit.get("excluded_conflicts"):
        pd.DataFrame(audit["excluded_conflicts"]).to_csv(
            destination / "excluded_conflicts.csv", index=False
        )

    audit.update(
        {
            "seed": seed,
            "near_duplicate_groups": int(frame.group_id.nunique()),
            "grouping": {
                "word_trigram_jaccard": 0.82,
                "containment": 0.95,
                "min_length_ratio": 0.5,
            },
            "strategy": (
                "Nested five-fold StratifiedGroupKFold, joint style/tone strata, first folds"
            ),
            "approximate_ratios": {"train": 0.64, "validation": 0.16, "test": 0.20},
            "splits": {},
        }
    )

    for name, part in frames.items():
        path = destination / f"{name}.csv"
        part.to_csv(path, index=False)
        audit["splits"][name] = {
            "rows": len(part),
            "sha256": fingerprint(path),
            "style": dict(Counter(part["style"])),
            "tone": dict(Counter(part["tone"])),
        }

    manifest_json = json.dumps(audit, indent=2)
    (destination / "manifest.json").write_text(manifest_json, encoding="utf-8")
    (destination / "corpus_audit.json").write_text(manifest_json, encoding="utf-8")

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "corpus_audit.json").write_text(manifest_json, encoding="utf-8")
    (reports_dir / "corpus_audit.md").write_text(generate_audit_markdown(audit), encoding="utf-8")
    (reports_dir / "data_splits.md").write_text(generate_splits_markdown(audit), encoding="utf-8")

    return frames, audit


def load_splits(
    directory: Path = PREPARED_DATA_DIR,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load and verify reproducible splits from ``directory`` against ``manifest.json``."""
    manifest_file = directory / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}

    for name, info in manifest["splits"].items():
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split file missing: {path}")
        if fingerprint(path) != info["sha256"]:
            raise ValueError(
                f"Split fingerprint mismatch for {path}. Expected {info['sha256']}, "
                f"got {fingerprint(path)}. Re-run data preparation before training."
            )
        frames[name] = pd.read_csv(path, keep_default_na=False)

    verify_splits(frames)
    return frames, manifest


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


def generate_splits_markdown(manifest: dict[str, Any]) -> str:
    """Generate a Markdown report detailing split sizes, distributions, and hashes."""
    total_rows = manifest.get("clean_rows", 998)
    splits = manifest.get("splits", {})

    lines: list[str] = [
        "# Data Splits and Leakage Prevention Report",
        "",
        "## Splitting Strategy",
        "",
        "- **Method:** Nested 5-fold `StratifiedGroupKFold` on combined `style__tone` strata.",
        (
            "- **Near-duplicate grouping:** Word trigram Jaccard similarity ≥ 0.82 "
            "or containment ≥ 0.95."
        ),
        f"- **Unique groups:** {manifest.get('near_duplicate_groups', 'N/A')} groups "
        f"across {total_rows} clean texts.",
        (
            f"- **Random seed:** `{manifest.get('seed', SEED)}` "
            "(outer fold seed=42, inner fold seed=43)."
        ),
        (
            "- **Disjointness:** Zero overlap between splits for "
            "`sample_id`, `group_id`, and normalized text."
        ),
        "",
        "## Split Summary",
        "",
        "| Split | Rows | Percentage | SHA-256 Hash |",
        "|---|---:|---:|---|",
    ]

    for name in ["train", "validation", "test"]:
        info = splits.get(name, {})
        rows = info.get("rows", 0)
        pct = (rows / total_rows * 100) if total_rows else 0
        sha = info.get("sha256", "N/A")
        lines.append(f"| **{name.capitalize()}** | {rows} | {pct:.1f}% | `{sha}` |")

    lines.extend(
        [
            "",
            "## Class Distributions Across Splits",
            "",
            "### Writing Styles",
            "",
            "| Style | Train | Validation | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for style in STYLES:
        tr = splits.get("train", {}).get("style", {}).get(style, 0)
        va = splits.get("validation", {}).get("style", {}).get(style, 0)
        te = splits.get("test", {}).get("style", {}).get(style, 0)
        lines.append(f"| `{style}` | {tr} | {va} | {te} | {tr + va + te} |")

    lines.extend(
        [
            "",
            "### Communicative Tones",
            "",
            "| Tone | Train | Validation | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for tone in TONES:
        tr = splits.get("train", {}).get("tone", {}).get(tone, 0)
        va = splits.get("validation", {}).get("tone", {}).get(tone, 0)
        te = splits.get("test", {}).get("tone", {}).get(tone, 0)
        lines.append(f"| `{tone}` | {tr} | {va} | {te} | {tr + va + te} |")

    lines.append("")
    return "\n".join(lines)


def build_and_audit(
    source_dir: Path = RAW_DATASET_DIR,
    raw_output: Path | None = RAW_DATASET_CSV,
    prepared_dir: Path = PREPARED_DATA_DIR,
    reports_dir: Path = CORPUS_AUDIT_JSON.parent,
    recompile_from_files: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Execute end-to-end dataset compilation, audit, and split preparation."""
    if recompile_from_files or raw_output is None or not raw_output.is_file():
        raw_df = compile_raw_corpus(source_dir)
        if raw_output is not None:
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(raw_output, index=False, encoding="utf-8")
        source: Path | pd.DataFrame = raw_df
    else:
        source = raw_output

    frames, manifest = prepare_splits(
        source=source,
        destination=prepared_dir,
        reports_dir=reports_dir,
    )
    return frames, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DATASET_CSV, help="Source dataset CSV")
    parser.add_argument(
        "--prepared-dir", type=Path, default=PREPARED_DATA_DIR, help="Directory for split CSVs"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=CORPUS_AUDIT_JSON.parent,
        help="Directory for audit and split reports",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for splitting")
    args = parser.parse_args()

    frames, manifest = prepare_splits(
        source=args.source,
        destination=args.prepared_dir,
        reports_dir=args.reports_dir,
        seed=args.seed,
    )
    print(
        f"Prepared {manifest['clean_rows']} samples into "
        f"train ({len(frames['train'])}), validation ({len(frames['validation'])}), "
        f"test ({len(frames['test'])}) -> {args.prepared_dir}"
    )
    print(f"Manifest written to {args.prepared_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
