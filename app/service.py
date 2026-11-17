"""Reusable calibrated prediction service shared by CLI, API, and UI."""

from __future__ import annotations

import math
import os
import threading
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from app.bundles import bundle_directory, verify_bundle
from app.config import LANGUAGE, MAX_BATCH_SIZE, MAX_INPUT_CHARS, MIN_INPUT_CHARS, STYLES, TONES
from app.input_validation import validate_batch, validate_batch_size
from app.prediction import NeuralPredictor
from app.registry import DEFAULT_REGISTRY, ActiveModelRegistry


class PredictionService:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        *,
        registry_path: str | Path = DEFAULT_REGISTRY,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.15,
        unknown_threshold: float = 0.30,
    ) -> None:
        thresholds = (confidence_threshold, margin_threshold, unknown_threshold)
        if any(
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not math.isfinite(x)
            or not 0 <= x <= 1
            for x in thresholds
        ):
            raise ValueError("Warning thresholds must be finite numbers between zero and one.")
        root = (
            ActiveModelRegistry(registry_path).resolve()
            if checkpoint_path is None
            else bundle_directory(checkpoint_path)
        )
        # The public service requires a complete, checksummed, calibrated bundle.
        self.bundle = verify_bundle(root)
        self.predictor = NeuralPredictor(root, device=device)
        if self.predictor.styles != STYLES or self.predictor.tones != TONES:
            raise ValueError("Saved labels and their order must match the project label sets.")
        self.tokenizer = self.predictor.tokenizer
        expected_kind = "transformer" if self.predictor.model_name == "distilbert" else "vocabulary"
        if self.tokenizer.kind != expected_kind:
            raise ValueError("Tokenizer kind is incompatible with the selected architecture.")
        if type(self.tokenizer.max_length) is not int or self.tokenizer.max_length < 1:
            raise ValueError("Saved max_length must be a positive integer.")
        if self.tokenizer.kind == "vocabulary":
            tokens = self.tokenizer.tokens
            if len(tokens) != len(set(tokens)) or tokens[:2] != ["<pad>", "<unk>"]:
                raise ValueError("Saved vocabulary must have unique tokens and valid special IDs.")
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.unknown_threshold = unknown_threshold
        self._lock = threading.Lock()

    def info(self) -> dict:
        return {
            "model": self.predictor.model_name,
            "language": LANGUAGE,
            "styles": list(self.predictor.styles),
            "tones": list(self.predictor.tones),
            "max_length": self.tokenizer.max_length,
            "tokenizer_kind": self.tokenizer.kind,
            "device": str(self.predictor.device),
            "temperatures": dict(self.predictor.temperatures),
            "weights_sha256": self.bundle["weights_sha256"],
            "input_limits": {
                "min_characters": MIN_INPUT_CHARS,
                "max_characters": MAX_INPUT_CHARS,
                "max_batch_size": MAX_BATCH_SIZE,
            },
            "warning_thresholds": {
                "confidence": self.confidence_threshold,
                "margin": self.margin_threshold,
                "unknown_ratio": self.unknown_threshold,
            },
        }

    def _token_information(self, text: str) -> dict:
        encoded = self.tokenizer.encode(text)
        if self.tokenizer.kind == "transformer":
            backend = self.tokenizer.tokenizer
            full_ids = backend(text, truncation=False, verbose=False)["input_ids"]
            unknown_id = backend.unk_token_id
            # UNK is itself listed as a special token; keep it in the denominator.
            excluded = set(backend.all_special_ids) - {unknown_id}
            content_ids = [token for token in encoded if token not in excluded]
            total = len(full_ids)
        else:
            total = self.tokenizer.inspect(text)["token_count"]
            unknown_id = self.tokenizer.unk_id
            content_ids = encoded
        unknown = content_ids.count(unknown_id) if unknown_id is not None else 0
        return {
            "character_count": len(text),
            "token_count": total,
            "used_token_count": len(encoded),
            "max_length": self.tokenizer.max_length,
            "truncated": total > self.tokenizer.max_length,
            "unknown_token_count": unknown,
            "unknown_ratio": unknown / len(content_ids) if content_ids else 0.0,
        }

    def _warnings(self, text: str, token_info: dict, scores: dict) -> list[dict]:
        warnings = []
        if token_info["truncated"]:
            warnings.append(
                {
                    "code": "truncated",
                    "task": None,
                    "message": f"Input was truncated to {self.tokenizer.max_length} tokens.",
                }
            )
        reasons = []
        if token_info["unknown_ratio"] > self.unknown_threshold:
            reasons.append(
                f"{token_info['unknown_ratio']:.0%} of retained content tokens are unknown"
            )
        letters = [character for character in text if character.isalpha()]
        if not letters:
            reasons.append("the message contains no alphabetic characters")
        elif (
            sum("LATIN" not in unicodedata.name(character, "") for character in letters)
            / len(letters)
            > 0.5
        ):
            reasons.append("most letters are outside the Latin alphabet used by the English model")
        if reasons:
            warnings.append(
                {
                    "code": "unfamiliar_input",
                    "task": None,
                    "message": "Input may be unfamiliar: " + "; ".join(reasons) + ".",
                }
            )
        for task in ("style", "tone"):
            ranked = sorted(scores[task].values(), reverse=True)
            if (
                ranked[0] < self.confidence_threshold
                or ranked[0] - ranked[1] < self.margin_threshold
            ):
                warnings.append(
                    {
                        "code": "uncertain_prediction",
                        "task": task,
                        "message": f"The {task} prediction is uncertain; inspect the class scores.",
                    }
                )
        return warnings

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[dict]:
        texts = validate_batch(texts)
        batch_size = validate_batch_size(batch_size)
        # Validate the whole request before tokenizing or running the model.
        # A lock also protects mutable fast-tokenizer truncation settings.
        with self._lock:
            token_info = [self._token_information(text) for text in texts]
            probabilities = self.predictor.predict_proba(texts, batch_size=batch_size)
        results = []
        for text, tokens, scores in zip(texts, token_info, probabilities, strict=True):
            result = {"model": self.predictor.model_name, **tokens}
            for task, labels in (("style", STYLES), ("tone", TONES)):
                values = scores[task]
                if (
                    tuple(values) != labels
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values.values())
                    or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6)
                ):
                    raise RuntimeError("Model produced invalid class probabilities.")
                label = max(values, key=values.get)
                result.update(
                    {task: label, f"{task}_confidence": values[label], f"{task}_scores": values}
                )
            result["warnings"] = self._warnings(text, tokens, scores)
            results.append(result)
        return results


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Load once per process; restart consumers after changing the active model."""
    return PredictionService(
        checkpoint_path=os.getenv("TSD_CHECKPOINT") or None,
        registry_path=os.getenv("TSD_REGISTRY") or DEFAULT_REGISTRY,
    )
