from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from app.bilstm import BiLSTMAttention
from app.config import STYLES, TONES, VOCAB_PAD_ID
from app.distilbert import DistilBERTClassifier


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        dropout: float = 0.5,
        pad_id: int = VOCAB_PAD_ID,
        num_styles: int = len(STYLES),
        num_tones: int = len(TONES),
    ) -> None:
        super().__init__()
        if min(vocab_size, embedding_dim, num_filters, num_styles, num_tones) <= 0:
            raise ValueError("Vocabulary, embedding, filter, and class sizes must be positive.")
        self.kernel_sizes = tuple(kernel_sizes)
        if not self.kernel_sizes or any(k <= 0 for k in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers.")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be an index inside the vocabulary.")

        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in self.kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        feature_dim = num_filters * len(self.kernel_sizes)
        self.style_head = nn.Linear(feature_dim, num_styles)
        self.tone_head = nn.Linear(feature_dim, num_tones)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return pooled features [B, num_filters * number_of_kernels]."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")
        token_mask = input_ids.ne(self.pad_id)
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != input_ids.shape[0]:
                raise ValueError("lengths must have shape [batch_size].")
            if lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("lengths must contain integer token counts.")
            lengths = lengths.to(input_ids.device)
            if ((lengths < 0) | (lengths > input_ids.shape[1])).any():
                raise ValueError("lengths must lie between zero and sequence_length.")
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            token_mask = token_mask & (positions.unsqueeze(0) < lengths.unsqueeze(1))

        # Support direct calls with L < 5, independently of Collator.min_length.
        extra = max(0, max(self.kernel_sizes) - input_ids.shape[1])
        if extra:
            input_ids = F.pad(input_ids, (0, extra), value=self.pad_id)
            token_mask = F.pad(token_mask, (0, extra), value=False)

        embeddings = self.embedding(input_ids)
        embeddings = embeddings.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        channels = embeddings.transpose(1, 2)  # [B, embedding_dim, L]
        pooled_features = []
        for kernel_size, conv in zip(self.kernel_sizes, self.convs, strict=True):
            features = F.relu(conv(channels))  # [B, num_filters, L-k+1]
            valid = token_mask.unfold(1, kernel_size, 1).all(dim=-1)
            features = features.masked_fill(~valid.unsqueeze(1), torch.finfo(features.dtype).min)
            pooled = features.amax(dim=-1)
            pooled = torch.where(valid.any(dim=1, keepdim=True), pooled, 0.0)
            pooled_features.append(pooled)
        return torch.cat(pooled_features, dim=1)

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        features = self.dropout(self.encode(input_ids, lengths))
        return {
            "style_logits": self.style_head(features),
            "tone_logits": self.tone_head(features),
        }


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "textcnn": TextCNN,
    "bilstm": BiLSTMAttention,
    "distilbert": DistilBERTClassifier,
}


def build_model(model_name: str, **kwargs: Any) -> nn.Module:
    """Construct a model implementing forward(input_ids, lengths) -> logits dict."""
    name = model_name.strip().lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; available: {', '.join(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](**kwargs)
