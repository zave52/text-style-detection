from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from app.config import STYLES, TONES, VOCAB_PAD_ID


class BiLSTMAttention(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        attention_dim: int = 64,
        dropout: float = 0.3,
        pad_id: int = VOCAB_PAD_ID,
        num_styles: int = len(STYLES),
        num_tones: int = len(TONES),
    ) -> None:
        super().__init__()
        sizes = (
            vocab_size,
            embedding_dim,
            hidden_size,
            num_layers,
            attention_dim,
            num_styles,
            num_tones,
        )
        if any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError(
                "Vocabulary, layer, hidden, attention, and class sizes must be positive."
            )
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be an index inside the vocabulary.")

        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(4 * hidden_size, num_styles)
        self.tone_head = nn.Linear(4 * hidden_size, num_tones)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
            # Set the input-side forget bias once, rather than in both LSTM biases.
            for name, parameter in self.lstm.named_parameters():
                if "bias_ih" in name:
                    parameter[hidden_size : 2 * hidden_size].fill_(1.0)

    def encode_with_attention(
        self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features [B, 4H] and attention weights [B, L].

        Inputs must contain contiguous tokens followed by right padding.
        Empty rows are represented by zeros and receive zero attention mass.
        """
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("input_ids must have non-empty shape [batch_size, sequence_length].")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs.")
        token_mask = input_ids.ne(self.pad_id)
        if lengths is None:
            lengths = token_mask.sum(dim=1)
        if lengths.ndim != 1 or lengths.shape[0] != input_ids.shape[0]:
            raise ValueError("lengths must have shape [batch_size].")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must contain integer token counts.")
        lengths = lengths.to(device=input_ids.device, dtype=torch.long)
        if ((lengths < 0) | (lengths > input_ids.shape[1])).any():
            raise ValueError("lengths must lie between zero and sequence_length.")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(token_mask, mask):
            raise ValueError(
                "Expected right padding consistent with lengths; no internal PAD tokens."
            )

        embeddings = self.embedding(input_ids).masked_fill(~mask.unsqueeze(-1), 0.0)
        embeddings = self.embedding_dropout(embeddings)
        packed = pack_padded_sequence(
            embeddings, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, _) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=input_ids.shape[1]
        )

        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1).masked_fill(~mask, 0.0)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)

        # h_n is restored to original batch order by the packed LSTM.
        # [-2] and [-1] are the forward/backward states of the last layer.
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        final_hidden = final_hidden.masked_fill(~lengths.gt(0).unsqueeze(1), 0.0)
        return torch.cat((context, final_hidden), dim=1), weights

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_with_attention(input_ids, lengths)[0]

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        features = self.dropout(self.encode(input_ids, lengths))
        return {
            "style_logits": self.style_head(features),
            "tone_logits": self.tone_head(features),
        }