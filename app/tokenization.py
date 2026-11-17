"""Tokenization, vocabulary management, and minibatch collation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from app.config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_LENGTH,
    SEED,
    STYLE_TO_ID,
    TONE_TO_ID,
    VOCAB_MAX_SIZE,
    VOCAB_PAD_ID,
    VOCAB_UNK_ID,
    normalize_style,
    normalize_tone,
)

# Matches alphanumeric words with internal apostrophes, or individual punctuation/symbols.
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into words and punctuation, preserving capitalization signals."""
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text):
        if len(token) > 1 and token.isupper() and any(c.isalpha() for c in token):
            tokens.append("<upper>")
        tokens.append(token.casefold().replace("’", "'"))
    return tokens


class Vocabulary:
    """Frequency-ranked word vocabulary with truncation and unknown token handling."""

    pad_id: int = VOCAB_PAD_ID
    unk_id: int = VOCAB_UNK_ID
    kind: str = "vocabulary"

    def __init__(self, tokens: list[str], max_length: int = DEFAULT_MAX_LENGTH) -> None:
        self.tokens = tokens
        self.lookup = {token: i for i, token in enumerate(tokens)}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        max_size: int = VOCAB_MAX_SIZE,
        min_frequency: int = 1,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> Vocabulary:
        """Fit vocabulary on training texts only."""
        counts = Counter(token for text in texts for token in tokenize(text))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        tokens = ["<pad>", "<unk>"]
        tokens.extend(t for t in ordered if counts[t] >= min_frequency and t not in tokens)
        return cls(tokens[:max_size], max_length=max_length)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs truncated to max_length."""
        token_ids = [self.lookup.get(t, self.unk_id) for t in tokenize(text)[: self.max_length]]
        return token_ids or [self.unk_id]

    def inspect(self, text: str) -> dict[str, Any]:
        """Return token count, truncation status, and unknown token ratio."""
        all_tokens = tokenize(text)
        selected = all_tokens[: self.max_length]
        unknown_count = sum(t not in self.lookup for t in selected)
        return {
            "token_count": len(all_tokens),
            "truncated": len(all_tokens) > self.max_length,
            "unknown_ratio": unknown_count / max(1, len(selected)),
        }

    def save(self, destination: Path | str) -> Path:
        """Persist vocabulary tokens and configuration to disk."""
        path = Path(destination)
        if path.is_dir() or path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "vocabulary.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                {
                    "tokens": self.tokens,
                    "max_length": self.max_length,
                    "kind": self.kind,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, source: Path | str) -> Vocabulary:
        """Load vocabulary from a directory or json file."""
        path = Path(source)
        if path.is_dir():
            path = path / "vocabulary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        info = json.loads(path.read_text(encoding="utf-8"))
        return cls(tokens=info["tokens"], max_length=info.get("max_length", DEFAULT_MAX_LENGTH))


class TransformerTokenizer:
    """Wrapper around HuggingFace AutoTokenizer providing a uniform API."""

    kind: str = "transformer"

    def __init__(
        self,
        source: str | Path,
        max_length: int = DEFAULT_MAX_LENGTH,
        local_only: bool = False,
    ) -> None:
        from transformers import AutoTokenizer

        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer of at least two tokens.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(source),
            local_files_only=local_only,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Configured transformer tokenizer must have a pad_token_id")
        if max_length > self.tokenizer.model_max_length:
            raise ValueError("max_length exceeds the tokenizer's supported sequence length.")
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = max_length
        self.pad_id = self.tokenizer.pad_token_id
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs with truncation."""
        return self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]

    def inspect(self, text: str) -> dict[str, Any]:
        """Inspect token count, truncation status, and unknown token ratio."""
        ids: list[int] = self.tokenizer(text, truncation=False)["input_ids"]
        selected = ids[: self.max_length]
        unk_count = selected.count(self.tokenizer.unk_token_id)
        return {
            "token_count": len(ids),
            "truncated": len(ids) > self.max_length,
            "unknown_ratio": unk_count / max(1, len(selected)),
        }

    def save(self, destination: Path | str) -> None:
        """Save pretrained tokenizer files."""
        self.tokenizer.save_pretrained(str(destination))


class TextDataset(Dataset):
    """PyTorch Dataset yielding token IDs and style/tone label indices."""

    def __init__(
        self,
        data: pd.DataFrame | Sequence[str],
        tokenizer: Vocabulary | TransformerTokenizer,
    ) -> None:
        self.has_labels: bool = False
        self.rows: list[tuple[list[int], int | None, int | None]] = []

        if isinstance(data, pd.DataFrame):
            self.has_labels = "style" in data.columns and "tone" in data.columns
            for row in data.itertuples():
                token_ids = tokenizer.encode(row.text)
                if self.has_labels:
                    style_id = STYLE_TO_ID[normalize_style(row.style)]
                    tone_id = TONE_TO_ID[normalize_tone(row.tone)]
                    self.rows.append((token_ids, style_id, tone_id))
                else:
                    self.rows.append((token_ids, None, None))
        else:
            for text in data:
                self.rows.append((tokenizer.encode(text), None, None))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[int], int | None, int | None]:
        return self.rows[index]


class Collator:
    """Collates variable-length token sequences into padded minibatches.

    Applies dynamic padding up to batch maximum while guaranteeing min_length
    so CNN kernel operations never receive inputs smaller than their filter sizes.
    """

    def __init__(
        self,
        pad_id: int = VOCAB_PAD_ID,
        min_length: int = DEFAULT_MIN_LENGTH,
        include_attention_mask: bool = False,
    ) -> None:
        self.pad_id = pad_id
        self.min_length = min_length
        self.include_attention_mask = include_attention_mask

    def __call__(
        self,
        rows: Sequence[tuple[list[int], ...] | list[Any]],
    ) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot collate an empty batch of rows")

        # Normalize rows to (tokens, style, tone) structure
        normalized_rows: list[tuple[list[int], int | None, int | None]] = []
        for r in rows:
            if isinstance(r, (list, tuple)):
                if len(r) > 0 and isinstance(r[0], list):
                    tokens = r[0]
                    style = r[1] if len(r) > 1 else None
                    tone = r[2] if len(r) > 2 else None
                    normalized_rows.append((tokens, style, tone))
                elif len(r) > 0 and isinstance(r[0], int):
                    # Direct token list passed
                    normalized_rows.append((list(r), None, None))
                else:
                    normalized_rows.append(([], None, None))
            else:
                raise TypeError(f"Unsupported row format: {type(r)}")

        lengths = torch.tensor([len(r[0]) for r in normalized_rows], dtype=torch.long)
        batch_seq_len = max(self.min_length, int(lengths.max()))
        input_ids = torch.full((len(normalized_rows), batch_seq_len), self.pad_id, dtype=torch.long)

        for i, (tokens, _, _) in enumerate(normalized_rows):
            if tokens:
                input_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "lengths": lengths,
        }
        if self.include_attention_mask:
            batch["attention_mask"] = input_ids.ne(self.pad_id).long()

        first = normalized_rows[0]
        if first[1] is not None:
            batch["style_labels"] = torch.tensor([r[1] for r in normalized_rows], dtype=torch.long)
        if first[2] is not None:
            batch["tone_labels"] = torch.tensor([r[2] for r in normalized_rows], dtype=torch.long)

        return batch


def make_loader(
    data: pd.DataFrame | Sequence[str],
    tokenizer: Vocabulary | TransformerTokenizer,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = SEED,
) -> DataLoader:
    """Construct a reproducible PyTorch DataLoader with dynamic padding."""
    transformer = tokenizer.kind == "transformer"
    return DataLoader(
        TextDataset(data, tokenizer),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(
            tokenizer.pad_id,
            min_length=1 if transformer else DEFAULT_MIN_LENGTH,
            include_attention_mask=transformer,
        ),
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
