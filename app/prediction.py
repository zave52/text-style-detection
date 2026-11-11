from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from app.config import ARTIFACTS_DIR, MAX_BATCH_SIZE, MAX_INPUT_CHARS, resolve_device
from app.models import build_model
from app.tokenization import TransformerTokenizer, Vocabulary, make_loader
from app.training import model_inputs


class NeuralPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(resolve_device(device))
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.styles = tuple(checkpoint["styles"])
        self.tones = tuple(checkpoint["tones"])
        kwargs = checkpoint["model_kwargs"]
        if checkpoint.get("tokenizer_kind") == "transformer":
            config = kwargs.get("encoder_config")
            if not config:
                raise ValueError("A local encoder configuration is required for offline inference.")
            self.tokenizer = TransformerTokenizer(
                checkpoint_path.parent / "tokenizer",
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
            self.tokenizer = Vocabulary(checkpoint["vocab_tokens"], checkpoint["max_length"])
            self.vocab = self.tokenizer  # Compatibility with existing vocabulary-based callers.
            if kwargs["vocab_size"] != len(self.vocab) or kwargs["pad_id"] != self.vocab.pad_id:
                raise ValueError("Checkpoint vocabulary does not match model configuration.")
        if kwargs["num_styles"] != len(self.styles) or kwargs["num_tones"] != len(self.tones):
            raise ValueError("Checkpoint class labels do not match classification heads.")
        self.model = build_model(checkpoint["model_name"], **kwargs)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts: str | Sequence[str], batch_size: int = 32) -> list[dict[str, str]]:
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
            style_ids = outputs["style_logits"].argmax(dim=1).cpu().tolist()
            tone_ids = outputs["tone_logits"].argmax(dim=1).cpu().tolist()
            predictions.extend(
                {"style": self.styles[style], "tone": self.tones[tone]}
                for style, tone in zip(style_ids, tone_ids, strict=True)
            )
        return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="+", help="Quote each input text separately.")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS_DIR / "bilstm" / "best.pt")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    args = parser.parse_args()
    predictor = NeuralPredictor(args.checkpoint, device=args.device)
    print(json.dumps(predictor.predict(args.texts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()