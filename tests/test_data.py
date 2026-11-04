"""Tests for corpus compilation, validation, and audit pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import (
    RAW_DATASET_DIR,
    STYLES,
    TONES,
)
from app.data import (
    build_and_audit,
    compile_raw_corpus,
    read_corpus,
    text_key,
)


def test_compile_raw_corpus():
    df = compile_raw_corpus(RAW_DATASET_DIR)
    assert len(df) == 1000
    assert set(df.columns) == {"text", "style", "tone"}
    assert "literaly" in set(df["style"])
    assert df["text"].str.len().gt(0).all()


def test_read_corpus_audit_on_source_files():
    raw_df = compile_raw_corpus(RAW_DATASET_DIR)
    clean_df, audit = read_corpus(raw_df)

    assert audit["source_rows"] == 1000
    assert audit["clean_rows"] == 998
    assert audit["conflicting_groups_removed"] == 1
    assert audit["conflicting_rows_removed"] == 2
    assert audit["label_aliases_fixed"] == 200
    assert audit["empty_removed"] == 0
    assert audit["duplicates_removed"] == 0

    # Ensure alias was normalized and all 5 styles/tones exist
    assert "literaly" not in set(clean_df["style"])
    assert set(clean_df["style"]) == set(STYLES)
    assert set(clean_df["tone"]) == set(TONES)

    # Check excluded records
    conflicts = audit["excluded_conflicts"]
    assert len(conflicts) == 2
    assert {c["text"] for c in conflicts} == {"Respond immediately."}
    assert {c["style"] for c in conflicts} == {"business", "formal"}
    assert {c["tone"] for c in conflicts} == {"urgent"}

    # Sample ID must be unique 16-character hex strings
    assert len(clean_df["sample_id"].unique()) == 998
    assert clean_df["sample_id"].str.len().eq(16).all()


def test_read_corpus_missing_columns():
    bad_df = pd.DataFrame([{"text": "Hello world", "style": "formal"}])
    with pytest.raises(ValueError, match="missing required columns"):
        read_corpus(bad_df)


def test_read_corpus_unknown_labels():
    bad_df = pd.DataFrame(
        [
            {"text": "Hello world", "style": "sci-fi", "tone": "neutral"},
        ]
    )
    with pytest.raises(ValueError, match="Unknown style labels"):
        read_corpus(bad_df)


def test_read_corpus_filters_empty_text():
    rows = []
    for s in STYLES:
        for t in TONES:
            rows.append({"text": f"Sample for {s} and {t}", "style": s, "tone": t})
    rows.append({"text": "   ", "style": "formal", "tone": "neutral"})
    df = pd.DataFrame(rows)

    clean_df, audit = read_corpus(df)
    assert audit["empty_removed"] == 1
    assert len(clean_df) == 25


def test_read_corpus_deduplicates_identical_labels():
    rows = []
    for s in STYLES:
        for t in TONES:
            rows.append({"text": f"Sample for {s} and {t}", "style": s, "tone": t})
    # Add an exact duplicate with same label
    rows.append({"text": "Sample for formal and neutral", "style": "formal", "tone": "neutral"})
    df = pd.DataFrame(rows)

    clean_df, audit = read_corpus(df)
    assert audit["duplicates_removed"] == 1
    assert audit["conflicting_rows_removed"] == 0
    assert len(clean_df) == 25


def test_build_and_audit_creates_all_files(tmp_path: Path):
    raw_csv = tmp_path / "raw" / "dataset.csv"
    prepared_dir = tmp_path / "prepared"
    reports_dir = tmp_path / "reports"

    clean_df, audit = build_and_audit(
        source_dir=RAW_DATASET_DIR,
        raw_output=raw_csv,
        prepared_dir=prepared_dir,
        reports_dir=reports_dir,
    )

    assert raw_csv.is_file()
    assert (prepared_dir / "dataset.csv").is_file()
    assert (prepared_dir / "excluded_conflicts.csv").is_file()
    assert (reports_dir / "corpus_audit.json").is_file()
    assert (reports_dir / "corpus_audit.md").is_file()

    # Check report content
    report_data = json.loads((reports_dir / "corpus_audit.json").read_text(encoding="utf-8"))
    assert report_data["clean_rows"] == 998

    md_content = (reports_dir / "corpus_audit.md").read_text(encoding="utf-8")
    assert "# Corpus Audit Report" in md_content
    assert "Respond immediately." in md_content


def test_text_key_normalizes_whitespace_and_case():
    assert text_key("  Hello   WORLD! \n") == "hello world!"
    assert text_key("Respond immediately.") == "respond immediately."
