"""Tests for corpus compilation, validation, audit, and splitting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import (
    MANIFEST_JSON,
    PREPARED_DATA_DIR,
    RAW_DATASET_DIR,
    STYLES,
    TONES,
)
from app.data import (
    build_and_audit,
    compile_raw_corpus,
    load_splits,
    prepare_splits,
    read_corpus,
    similarity_groups,
    text_key,
    verify_splits,
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

    frames, audit = build_and_audit(
        source_dir=RAW_DATASET_DIR,
        raw_output=raw_csv,
        prepared_dir=prepared_dir,
        reports_dir=reports_dir,
        recompile_from_files=True,
    )

    assert raw_csv.is_file()
    assert (prepared_dir / "dataset.csv").is_file()
    assert (prepared_dir / "excluded_conflicts.csv").is_file()
    assert (prepared_dir / "train.csv").is_file()
    assert (prepared_dir / "validation.csv").is_file()
    assert (prepared_dir / "test.csv").is_file()
    assert (prepared_dir / "manifest.json").is_file()
    assert (reports_dir / "corpus_audit.json").is_file()
    assert (reports_dir / "corpus_audit.md").is_file()
    assert (reports_dir / "data_splits.md").is_file()

    # Check report content
    report_data = json.loads((reports_dir / "corpus_audit.json").read_text(encoding="utf-8"))
    assert report_data["clean_rows"] == 998
    assert "splits" in report_data

    md_content = (reports_dir / "corpus_audit.md").read_text(encoding="utf-8")
    assert "# Corpus Audit Report" in md_content
    assert "Respond immediately." in md_content

    splits_md = (reports_dir / "data_splits.md").read_text(encoding="utf-8")
    assert "# Data Splits and Leakage Prevention Report" in splits_md
    assert "Train" in splits_md


def test_text_key_normalizes_whitespace_and_case():
    assert text_key("  Hello   WORLD! \n") == "hello world!"
    assert text_key("Respond immediately.") == "respond immediately."


def test_similarity_groups_clusters_near_duplicates():
    p1 = "The research committee thoroughly reviewed all submitted experimental results and data."
    p2 = (
        "The research committee thoroughly reviewed all submitted experimental results and data. "
        "Further validation steps will be planned."
    )
    p3 = "Completely distinct sentence with totally different words and subject matter entirely."

    groups = similarity_groups([p1, p2, p3])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_verify_splits_detects_leakage():
    df_train = pd.DataFrame(
        [
            {
                "text": "Text A",
                "style": "formal",
                "tone": "neutral",
                "sample_id": "1",
                "group_id": 1,
            },
            {
                "text": "Text B",
                "style": "formal",
                "tone": "neutral",
                "sample_id": "2",
                "group_id": 2,
            },
        ]
    )
    df_val = pd.DataFrame(
        [
            {
                "text": "Text C",
                "style": "formal",
                "tone": "neutral",
                "sample_id": "3",
                "group_id": 3,
            },
        ]
    )

    # Valid split passes
    verify_splits({"train": df_train, "val": df_val})

    # Overlapping sample_id raises ValueError
    leaky_sample = pd.DataFrame(
        [
            {
                "text": "Text X",
                "style": "formal",
                "tone": "neutral",
                "sample_id": "1",
                "group_id": 4,
            },
        ]
    )
    with pytest.raises(ValueError, match="sample_id overlap"):
        verify_splits({"train": df_train, "val": leaky_sample})

    # Overlapping group_id raises ValueError
    leaky_group = pd.DataFrame(
        [
            {
                "text": "Text Y",
                "style": "formal",
                "tone": "neutral",
                "sample_id": "5",
                "group_id": 1,
            },
        ]
    )
    with pytest.raises(ValueError, match="group_id overlap"):
        verify_splits({"train": df_train, "val": leaky_group})

    # Duplicate text across splits raises ValueError
    leaky_text = pd.DataFrame(
        [
            {
                "text": "  text a  ",
                "style": "formal",
                "tone": "neutral",
                "sample_id": "6",
                "group_id": 6,
            },
        ]
    )
    with pytest.raises(ValueError, match="Duplicate text crosses split boundary"):
        verify_splits({"train": df_train, "val": leaky_text})


def test_load_splits_and_manifest_verification():
    frames, manifest = load_splits(PREPARED_DATA_DIR)
    assert set(frames.keys()) == {"train", "validation", "test"}
    assert sum(len(f) for f in frames.values()) == 998
    assert len(frames["train"]) == 632
    assert len(frames["validation"]) == 158
    assert len(frames["test"]) == 208

    # Verify no leakage
    assert not (set(frames["train"].group_id) & set(frames["test"].group_id))
    assert not (set(frames["train"].group_id) & set(frames["validation"].group_id))
    assert not (set(frames["validation"].group_id) & set(frames["test"].group_id))

    # Verify manifest file matches
    expected_manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["sha256"] == expected_manifest["splits"]["train"]["sha256"]


def test_load_splits_fails_on_tampered_file(tmp_path: Path):
    # Setup mock prepared directory
    mock_dir = tmp_path / "mock_prepared"
    prepare_splits(source=PREPARED_DATA_DIR / "dataset.csv", destination=mock_dir)

    # Tamper with test.csv
    test_csv = mock_dir / "test.csv"
    test_csv.write_text(test_csv.read_text() + "\ntampered,row,here,123,456\n")

    with pytest.raises(ValueError, match="Split fingerprint mismatch"):
        load_splits(mock_dir)
