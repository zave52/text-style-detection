"""Tests for tokenization, vocabulary management, and DataLoader collation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from app.config import (
    DEFAULT_MIN_LENGTH,
    SEED,
    STYLE_TO_ID,
    TONE_TO_ID,
    VOCAB_PAD_ID,
    VOCAB_UNK_ID,
)
from app.tokenization import (
    Collator,
    TextDataset,
    TransformerTokenizer,
    Vocabulary,
    make_loader,
    tokenize,
)


def test_tokenize_preserves_punctuation_and_capitalization() -> None:
    text = "PLEASE send the revised report today! It's URGENT."
    tokens = tokenize(text)
    assert "<upper>" in tokens
    assert "please" in tokens
    assert "!" in tokens
    assert "." in tokens
    assert "it's" in tokens
    assert tokens.count("<upper>") == 2  # PLEASE and URGENT


def test_tokenize_normalizes_curly_apostrophes() -> None:
    text = "Don’t worry, it’s fine."
    tokens = tokenize(text)
    assert "don't" in tokens
    assert "it's" in tokens


def test_tokenize_empty_and_single_characters() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("I am a student.") == ["i", "am", "a", "student", "."]


def test_vocabulary_fit_uses_training_texts_only() -> None:
    train_texts = ["Training sample one.", "Training sample two."]
    vocab = Vocabulary.fit(train_texts)

    assert vocab.tokens[0] == "<pad>"
    assert vocab.tokens[1] == "<unk>"
    assert vocab.pad_id == VOCAB_PAD_ID
    assert vocab.unk_id == VOCAB_UNK_ID

    encoded = vocab.encode("sample")
    assert encoded != [VOCAB_UNK_ID]

    encoded_unknown = vocab.encode("validationonlyword")
    assert encoded_unknown == [VOCAB_UNK_ID]


def test_vocabulary_fit_frequency_and_size_limits() -> None:
    texts = ["apple apple apple banana banana orange"]
    vocab = Vocabulary.fit(texts, min_frequency=2)
    assert "apple" in vocab.lookup
    assert "banana" in vocab.lookup
    assert "orange" not in vocab.lookup

    capped_vocab = Vocabulary.fit(texts, max_size=3)
    assert len(capped_vocab) == 3


def test_vocabulary_encode_and_truncation() -> None:
    vocab = Vocabulary.fit(["alpha beta gamma delta"], max_length=2)
    encoded = vocab.encode("alpha beta gamma delta")
    assert len(encoded) == 2

    empty_encoded = vocab.encode("")
    assert empty_encoded == [VOCAB_UNK_ID]


def test_vocabulary_inspect() -> None:
    vocab = Vocabulary.fit(["alpha beta"], max_length=3)
    info = vocab.inspect("alpha beta gamma delta")
    assert info["token_count"] == 4
    assert info["truncated"] is True
    assert 0.0 < info["unknown_ratio"] < 1.0


def test_vocabulary_save_and_load_roundtrip(tmp_path: Path) -> None:
    vocab = Vocabulary.fit(["Hello WORLD!"])
    saved_path = vocab.save(tmp_path)
    loaded = Vocabulary.load(saved_path)

    assert loaded.tokens == vocab.tokens
    assert loaded.max_length == vocab.max_length
    assert loaded.encode("Hello WORLD!") == vocab.encode("Hello WORLD!")


def test_text_dataset_labeled_and_unlabeled() -> None:
    df = pd.DataFrame(
        {
            "text": ["First sample.", "Second sample."],
            "style": ["formal", "literaly"],
            "tone": ["neutral", "urgent"],
        }
    )
    vocab = Vocabulary.fit(df["text"])
    dataset = TextDataset(df, vocab)

    assert len(dataset) == 2
    assert dataset.has_labels is True

    item0 = dataset[0]
    assert len(item0[0]) > 0
    assert item0[1] == STYLE_TO_ID["formal"]
    assert item0[2] == TONE_TO_ID["neutral"]

    item1 = dataset[1]
    assert item1[1] == STYLE_TO_ID["literary"]  # Normalized alias
    assert item1[2] == TONE_TO_ID["urgent"]

    unlabeled_dataset = TextDataset(["Unlabeled text."], vocab)
    assert unlabeled_dataset.has_labels is False
    assert unlabeled_dataset[0][1] is None
    assert unlabeled_dataset[0][2] is None


def test_collator_variable_lengths_and_min_length() -> None:
    collator = Collator(pad_id=VOCAB_PAD_ID, min_length=DEFAULT_MIN_LENGTH)

    rows = [
        ([10, 11], 0, 1),
        ([20, 21, 22, 23, 24, 25, 26], 2, 3),
    ]
    batch = collator(rows)

    assert batch["input_ids"].shape == (2, 7)
    assert torch.equal(batch["lengths"], torch.tensor([2, 7]))
    assert torch.equal(batch["style_labels"], torch.tensor([0, 2]))
    assert torch.equal(batch["tone_labels"], torch.tensor([1, 3]))

    short_rows = [([10], 0, 1)]
    short_batch = collator(short_rows)
    assert short_batch["input_ids"].shape == (1, DEFAULT_MIN_LENGTH)
    assert short_batch["input_ids"][0, 0].item() == 10
    assert (short_batch["input_ids"][0, 1:] == VOCAB_PAD_ID).all()


def test_collator_direct_tokens_without_labels() -> None:
    collator = Collator()
    rows = [[5, 6, 7], [8]]
    batch = collator(rows)
    assert "style_labels" not in batch
    assert "tone_labels" not in batch
    assert batch["input_ids"].shape == (2, DEFAULT_MIN_LENGTH)


def test_collator_empty_rows_raises() -> None:
    collator = Collator()
    with pytest.raises(ValueError, match="Cannot collate an empty batch"):
        collator([])


def test_make_loader_deterministic() -> None:
    df = pd.DataFrame(
        {
            "text": [f"Text sample number {i}" for i in range(10)],
            "style": ["formal"] * 10,
            "tone": ["neutral"] * 10,
        }
    )
    vocab = Vocabulary.fit(df["text"])

    loader1 = make_loader(df, vocab, batch_size=4, shuffle=True, seed=SEED)
    loader2 = make_loader(df, vocab, batch_size=4, shuffle=True, seed=SEED)

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    assert torch.equal(batch1["input_ids"], batch2["input_ids"])
    assert torch.equal(batch1["lengths"], batch2["lengths"])


def test_transformer_tokenizer_wrapper() -> None:
    wrapper = object.__new__(TransformerTokenizer)
    wrapper.kind = "transformer"
    wrapper.max_length = 64
    wrapper.pad_id = 0

    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.unk_token_id = 1
    mock_tok.side_effect = None
    mock_tok.return_value = {"input_ids": [101, 200, 102]}

    wrapper.tokenizer = mock_tok

    ids = wrapper.encode("Test string")
    assert ids == [101, 200, 102]

    inspection = wrapper.inspect("Test string")
    assert inspection["token_count"] == 3
    assert inspection["truncated"] is False
    assert inspection["unknown_ratio"] == 0.0
