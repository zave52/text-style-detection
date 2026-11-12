from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from app.bundles import read_json, verify_bundle
from app.calibration import temperatures_from
from app.config import MAX_BATCH_SIZE, MAX_INPUT_CHARS, resolve_device
from app.models import build_model
from app.tokenization import TransformerTokenizer, Vocabulary, make_loader
from app.training import model_inputs


class NeuralPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        *,
        registry_path: str | Path | None = None,
    ) -> None:
        if checkpoint_path is None:
            from app.registry import DEFAULT_REGISTRY, ActiveModelRegistry

            checkpoint_path = ActiveModelRegistry(registry_path or DEFAULT_REGISTRY).resolve()

        self.device = torch.device(resolve_device(device))
        checkpoint_path = Path(checkpoint_path).resolve()

        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "best.pt"

        root = checkpoint_path.parent
        manifest = None

        if (root / "bundle.json").is_file():
            if checkpoint_path.name != "best.pt":
                raise ValueError("Bundle checkpoints must use best.pt.")
            manifest = verify_bundle(root)
        elif (root / "calibration.json").exists():
            raise ValueError("Calibrated checkpoint is missing bundle.json integrity information.")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        self.checkpoint_path = checkpoint_path
        self.model_name = checkpoint["model_name"]
        self.metadata = checkpoint.get("metadata", {})
        self.temperatures = {"style": 1.0, "tone": 1.0}

        if "calibration" in checkpoint:
            if manifest is None:
                raise ValueError("Calibrated checkpoint requires a verified bundle.json.")

            calibration = read_json(root / "calibration.json")

            if calibration != checkpoint["calibration"]:
                raise ValueError("Calibration metadata does not match the checkpoint.")

            self.temperatures = temperatures_from(calibration)

        if manifest is not None:
            labels = {key: checkpoint[key] for key in ("styles", "tones")}
            config = {key: checkpoint[key] for key in ("model_name", "model_kwargs", "max_length")}
            config["tokenizer_kind"] = checkpoint.get("tokenizer_kind", "vocabulary")

            if (
                "calibration" not in checkpoint
                or manifest["model_name"] != self.model_name
                or read_json(root / "labels.json") != labels
                or read_json(root / "model_config.json") != json.loads(json.dumps(config))
                or read_json(root / "training_metadata.json")
                != json.loads(json.dumps(self.metadata))
            ):
                raise ValueError("Bundle metadata does not match the checkpoint.")

        self.styles = tuple(checkpoint["styles"])
        self.tones = tuple(checkpoint["tones"])
        kwargs = checkpoint["model_kwargs"]

        if checkpoint.get("tokenizer_kind") == "transformer":
            config = kwargs.get("encoder_config")

            if not config:
                raise ValueError("A local encoder configuration is required for offline inference.")

            self.tokenizer = TransformerTokenizer(
                root / "tokenizer",
                max_length=checkpoint["max_length"],
                local_only=True,
            )

            if len(self.tokenizer.tokenizer) != config["vocab_size"]:
                raise ValueError("Saved tokenizer does not match the encoder vocabulary.")
            if self.tokenizer.pad_id != config["pad_token_id"]:
                raise ValueError("Saved tokenizer PAD ID does not match the encoder.")
            if self.tokenizer.max_length > config["max_position_embeddings"]:
                raise ValueError("Saved sequence length exceeds the encoder limit.")
        else:
            self.tokenizer = Vocabulary(
                checkpoint["vocab_tokens"],
                checkpoint["max_length"],
            )
            self.vocab = self.tokenizer

            if kwargs["vocab_size"] != len(self.vocab) or kwargs["pad_id"] != self.vocab.pad_id:
                raise ValueError("Checkpoint vocabulary does not match model configuration.")

            if manifest is not None:
                saved = Vocabulary.load(root)

                if saved.tokens != self.vocab.tokens or saved.max_length != self.vocab.max_length:
                    raise ValueError("Saved vocabulary does not match the checkpoint.")

        if kwargs["num_styles"] != len(self.styles) or kwargs["num_tones"] != len(self.tones):
            raise ValueError("Checkpoint class labels do not match classification heads.")

        self.model = build_model(checkpoint["model_name"], **kwargs)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()

    def predict(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
    ) -> list[dict[str, str]]:
        return [
            {task: max(row[task], key=row[task].get) for task in ("style", "tone")}
            for row in self.predict_proba(texts, batch_size=batch_size)
        ]

    @torch.inference_mode()
    def predict_proba(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
    ) -> list[dict[str, dict[str, float]]]:
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}.")

        texts = [texts] if isinstance(texts, str) else list(texts)

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Prediction inputs must be non-empty strings.")
        if any(len(text) > MAX_INPUT_CHARS for text in texts):
            raise ValueError(f"Each text must contain at most {MAX_INPUT_CHARS} characters.")

        self.model.eval()
        predictions = []

        for batch in make_loader(texts, self.tokenizer, batch_size=batch_size):
            inputs = {key: value.to(self.device) for key, value in model_inputs(batch).items()}
            outputs = self.model(**inputs)

            probabilities = {
                task: torch.softmax(
                    outputs[f"{task}_logits"].double() / self.temperatures[task],
                    dim=1,
                )
                .cpu()
                .tolist()
                for task in ("style", "tone")
            }

            predictions.extend(
                {
                    "style": dict(zip(self.styles, style, strict=True)),
                    "tone": dict(zip(self.tones, tone, strict=True)),
                }
                for style, tone in zip(
                    probabilities["style"],
                    probabilities["tone"],
                    strict=True,
                )
            )

        return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="+", help="Quote each input text separately.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Defaults to the active registry model.",
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Show calibrated probabilities.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    args = parser.parse_args()

    predictor = NeuralPredictor(
        args.checkpoint,
        device=args.device,
        registry_path=args.registry,
    )
    result = (
        predictor.predict_proba(args.texts) if args.probabilities else predictor.predict(args.texts)
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
