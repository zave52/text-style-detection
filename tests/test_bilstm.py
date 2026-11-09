import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from app.config import STYLES, TONES
from app.models import build_model
from app.prediction import NeuralPredictor
from app.tokenization import Collator, Vocabulary, make_loader
from app.training import multitask_loss, run_epoch


def small_model(**overrides):
    torch.manual_seed(42)
    kwargs = {
        "vocab_size": 32,
        "embedding_dim": 8,
        "hidden_size": 6,
        "attention_dim": 4,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return build_model("bilstm", **kwargs)


def sample_batch():
    return Collator()([([2, 3, 4], 0, 1), ([5, 6, 7, 8, 9, 10, 11], 2, 3), ([12], 4, 0)])


@pytest.mark.parametrize("num_layers", [1, 2])
def test_dimensions_and_attention_mask(num_layers):
    model = small_model(num_layers=num_layers).eval()
    batch = sample_batch()
    features, weights = model.encode_with_attention(batch["input_ids"], batch["lengths"])
    outputs = model(batch["input_ids"], batch["lengths"])
    assert features.shape == (3, 24)
    assert outputs["style_logits"].shape == (3, len(STYLES))
    assert outputs["tone_logits"].shape == (3, len(TONES))
    assert weights.shape == batch["input_ids"].shape
    assert torch.count_nonzero(weights[batch["input_ids"] == 0]) == 0
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_packing_and_both_final_hidden_states():
    model = small_model(num_layers=2).eval()
    batch = sample_batch()
    captured = {}

    def capture(_module, args, output):
        assert isinstance(args[0], PackedSequence)
        assert args[0].data.shape[0] == int(batch["lengths"].sum())
        captured["hidden"] = output[1][0]

    handle = model.lstm.register_forward_hook(capture)
    try:
        features = model.encode(batch["input_ids"], batch["lengths"])
    finally:
        handle.remove()
    hidden = captured["hidden"]
    torch.testing.assert_close(features[:, 12:], torch.cat((hidden[-2], hidden[-1]), dim=1))


def test_padding_and_batch_order_do_not_change_predictions():
    model = small_model(dropout=0.4, num_layers=2, pad_id=31).eval()
    rows = [[2, 3, 4], [5, 6, 7, 8, 9, 10, 11], [12]]
    batch = Collator(pad_id=31)(rows)
    together = model(batch["input_ids"], batch["lengths"])
    for index, tokens in enumerate(rows):
        alone = model(torch.tensor([tokens]))
        padded = model(torch.tensor([tokens + [31] * 20]))
        for key in alone:
            torch.testing.assert_close(alone[key][0], together[key][index], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(alone[key], padded[key], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("task", ["style", "tone"])
def test_gradients_reach_embeddings_both_directions_and_attention(task):
    model = small_model()
    batch = sample_batch()
    outputs = model(batch["input_ids"], batch["lengths"])
    F.cross_entropy(outputs[f"{task}_logits"], batch[f"{task}_labels"]).backward()
    head = getattr(model, f"{task}_head")
    for module in (model.embedding, model.lstm, model.attention, head):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
    other = model.tone_head if task == "style" else model.style_head
    assert all(parameter.grad is None for parameter in other.parameters())


@pytest.mark.parametrize("mixed", [False, True])
def test_empty_rows_have_zero_features_and_finite_gradients(mixed):
    model = small_model()
    batch = Collator()([([], 0, 1), ([2, 3] if mixed else [], 2, 3)])
    features, weights = model.encode_with_attention(batch["input_ids"], batch["lengths"])
    empty = batch["lengths"] == 0
    assert torch.count_nonzero(features[empty]) == 0
    assert torch.count_nonzero(weights[empty]) == 0
    loss = multitask_loss(model(batch["input_ids"], batch["lengths"]), batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("tokens", [[0, 2, 3], [2, 0, 3]])
def test_noncontiguous_tokens_are_rejected_instead_of_silently_truncated(tokens):
    with pytest.raises(ValueError, match="right padding"):
        small_model()(torch.tensor([tokens]))


@pytest.mark.parametrize("model_name", ["bilstm", "textcnn"])
def test_shared_training_checkpoint_and_prediction(tmp_path, model_name):
    torch.manual_seed(42)
    texts = ["Please send the financial report tomorrow.", "Hello!", "What a wonderful game today!"]
    frame = pd.DataFrame(
        {
            "text": texts,
            "style": ["business", "informal", "informal"],
            "tone": ["neutral", "friendly", "friendly"],
        }
    )
    vocab = Vocabulary.fit(texts)
    kwargs = {
        "vocab_size": len(vocab),
        "embedding_dim": 8,
        "dropout": 0.3,
        "pad_id": 0,
        "num_styles": len(STYLES),
        "num_tones": len(TONES),
    }
    kwargs.update(
        {"hidden_size": 6, "attention_dim": 4} if model_name == "bilstm" else {"num_filters": 4}
    )
    model = build_model(model_name, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    metrics = run_epoch(model, make_loader(frame, vocab, batch_size=2), optimizer=optimizer)
    assert metrics["loss"] > 0
    for name in ("embedding.weight", "style_head.weight", "tone_head.weight"):
        assert not torch.equal(before[name], dict(model.named_parameters())[name])
    assert torch.count_nonzero(model.embedding.weight[0]) == 0
    optimizer.zero_grad(set_to_none=True)
    run_epoch(model, make_loader(frame, vocab, batch_size=2))
    assert not model.training and all(p.grad is None for p in model.parameters())

    styles, tones = list(reversed(STYLES)), list(reversed(TONES))
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": kwargs,
            "state_dict": model.state_dict(),
            "styles": styles,
            "tones": tones,
            "vocab_tokens": vocab.tokens,
            "max_length": vocab.max_length,
        },
        checkpoint,
    )
    predictor = NeuralPredictor(checkpoint, device="cpu")
    batch = next(iter(make_loader(texts, vocab)))
    with torch.no_grad():
        outputs = model(batch["input_ids"], batch["lengths"])
        restored = predictor.model(batch["input_ids"], batch["lengths"])
    for key in outputs:
        torch.testing.assert_close(outputs[key], restored[key])
    expected = [
        {"style": styles[s], "tone": tones[t]}
        for s, t in zip(
            outputs["style_logits"].argmax(1).tolist(),
            outputs["tone_logits"].argmax(1).tolist(),
            strict=True,
        )
    ]
    assert predictor.predict(texts, batch_size=2) == expected
    assert predictor.predict(texts, batch_size=1) == expected
    assert predictor.predict(texts[0]) == expected[:1]
    assert predictor.predict([]) == []