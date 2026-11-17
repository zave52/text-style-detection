"""Shared input limits for the prediction service, API, and interface."""

from __future__ import annotations

from collections.abc import Sequence

from app.config import MAX_BATCH_SIZE, MAX_INPUT_CHARS, MIN_INPUT_CHARS


class PredictionInputError(ValueError):
    """A prediction request violates the public input contract."""


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise PredictionInputError("Each message must be a string.")
    if not text.strip():
        raise PredictionInputError("Messages must not be empty or contain only whitespace.")
    if not MIN_INPUT_CHARS <= len(text) <= MAX_INPUT_CHARS:
        raise PredictionInputError(
            f"Each message must contain {MIN_INPUT_CHARS}–{MAX_INPUT_CHARS} characters."
        )
    return text


def validate_batch(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise PredictionInputError("A batch must be a list or sequence of messages.")
    if not 1 <= len(texts) <= MAX_BATCH_SIZE:
        raise PredictionInputError(f"A batch must contain 1–{MAX_BATCH_SIZE} messages.")
    validated = []
    for index, text in enumerate(texts):
        try:
            validated.append(validate_text(text))
        except PredictionInputError as error:
            raise PredictionInputError(f"Message at index {index}: {error}") from error
    return validated


def validate_batch_size(batch_size: int) -> int:
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise PredictionInputError(f"batch_size must be an integer from 1 to {MAX_BATCH_SIZE}.")
    return batch_size
