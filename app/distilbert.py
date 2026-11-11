from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.config import STYLES, TONES
from app.tokenization import TransformerTokenizer

PRETRAINED_NAME = "distilbert/distilbert-base-cased"


class DistilBERTClassifier(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any] | None = None,
        pretrained_name: str = PRETRAINED_NAME,
        dropout: float = 0.2,
        num_styles: int = len(STYLES),
        num_tones: int = len(TONES),
        local_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import DistilBertConfig, DistilBertModel

        if min(num_styles, num_tones) <= 0:
            raise ValueError("Class counts must be positive.")
        if encoder_config is None:
            self.encoder = DistilBertModel.from_pretrained(
                pretrained_name, local_files_only=local_only
            )
        else:
            self.encoder = DistilBertModel(DistilBertConfig.from_dict(encoder_config))

        dim = self.encoder.config.dim
        self.pre_classifier = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(dim, num_styles)
        self.tone_head = nn.Linear(dim, num_tones)
        for layer in (self.pre_classifier, self.style_head, self.tone_head):
            nn.init.normal_(layer.weight, std=self.encoder.config.initializer_range)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("input_ids must have non-empty shape [batch_size, sequence_length].")
        if input_ids.shape[1] > self.encoder.config.max_position_embeddings:
            raise ValueError("Input exceeds the encoder's maximum sequence length.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.encoder.config.pad_token_id)
            if lengths is not None:
                positions = torch.arange(input_ids.shape[1], device=input_ids.device)
                attention_mask &= positions.unsqueeze(0) < lengths.to(input_ids.device).unsqueeze(1)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        attention_mask = attention_mask.to(input_ids.device)
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must contain only zero and one.")
        if not attention_mask[:, 0].bool().all():
            raise ValueError("The first (CLS) token must be unmasked in every row.")

        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask.long(), return_dict=True
        )
        features = self.dropout(
            self.activation(self.pre_classifier(output.last_hidden_state[:, 0]))
        )
        return {"style_logits": self.style_head(features), "tone_logits": self.tone_head(features)}

    def save_bundle(
        self,
        directory: str | Path,
        tokenizer: TransformerTokenizer,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save full encoder/head weights, config, tokenizer, labels, and input limit."""
        config = self.encoder.config
        if not 2 <= tokenizer.max_length <= config.max_position_embeddings:
            raise ValueError("Tokenizer length limit is incompatible with the encoder.")
        if len(tokenizer.tokenizer) != config.vocab_size or tokenizer.pad_id != config.pad_token_id:
            raise ValueError("Tokenizer vocabulary is incompatible with the encoder.")
        if self.style_head.out_features != len(STYLES) or self.tone_head.out_features != len(TONES):
            raise ValueError("Classification heads must match the project's label sets.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(directory)
        tokenizer.save(directory / "tokenizer")
        checkpoint = directory / "best.pt"
        torch.save(
            {
                "model_name": "distilbert",
                "model_kwargs": {
                    "encoder_config": config.to_dict(),
                    "dropout": self.dropout.p,
                    "num_styles": self.style_head.out_features,
                    "num_tones": self.tone_head.out_features,
                },
                "tokenizer_kind": "transformer",
                "max_length": tokenizer.max_length,
                "styles": list(STYLES),
                "tones": list(TONES),
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            checkpoint,
        )
        return checkpoint