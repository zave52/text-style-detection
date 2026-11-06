"""TextCNN dimensions, padding safety, independent gradients, and training integration."""

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from app.config import STYLES, TONES
from app.models import TextCNN, build_model
from app.tokenization import Collator, Vocabulary, make_loader
from app.training import multitask_loss, run_epoch


def small_model(**overrides):
    torch.manual_seed(42)
    kwargs = {"vocab_size": 32, "embedding_dim": 8, "num_filters": 4, "dropout": 0.0}
    kwargs.update(overrides)
    return build_model("textcnn", **kwargs)


@pytest.mark.parametrize("batch_size,length", [(1, 0), (1, 1), (2, 2), (3, 5), (4, 13)])
def test_output_dimensions_and_finite_values(batch_size, length):
    model = small_model().eval()
    ids = torch.randint(1, 32, (batch_size, length))
    outputs = model(ids)
    assert model.encode(ids).shape == (batch_size, 12)
    assert outputs["style_logits"].shape == (batch_size, len(STYLES))
    assert outputs["tone_logits"].shape == (batch_size, len(TONES))
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_padding_invariance_alone_and_in_a_longer_batch():
    model = small_model(pad_id=31).eval()
    tokens = [2, 3, 4, 5, 6, 7]
    alone = model(torch.tensor([tokens]), torch.tensor([len(tokens)]))
    batch = Collator(pad_id=31)([(tokens, None, None), (list(range(1, 22)), None, None)])
    together = model(batch["input_ids"], batch["lengths"])
    for key in alone:
        torch.testing.assert_close(alone[key][0], together[key][0], atol=1e-6, rtol=1e-5)

    # Without lengths, infer the mask from pad_id, including left padding.
    padded_ids = torch.tensor([[31, 31] + tokens + [31] * 10])
    padded = model(padded_ids)
    for key in alone:
        torch.testing.assert_close(alone[key], padded[key], atol=1e-6, rtol=1e-5)


def test_partial_and_padding_only_windows_cannot_win_pooling():
    model = small_model(embedding_dim=1, num_filters=1).eval()
    with torch.no_grad():
        model.embedding.weight.fill_(1.0)
        model.embedding.weight[0].fill_(-100.0)  # Even a corrupted PAD vector is ignored.
        for conv in model.convs:
            conv.weight.fill_(-1.0)
            conv.bias.fill_(1.5)
    # Full windows give ReLU(1.5-k)=0; a partial 1-token window would give 0.5,
    # while a padding-only window would give 1.5 without the pooling mask.
    ids = torch.tensor([[2, 3, 4, 5, 6] + [0] * 9, [2, 3] + [0] * 12, [0] * 14])
    torch.testing.assert_close(model.encode(ids), torch.zeros(3, 3))


def test_explicit_lengths_hide_nonpadding_tail_tokens():
    model = small_model().eval()
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    expected = model(ids)
    actual = model(torch.tensor([[2, 3, 4, 5, 6, 29, 30, 28]]), torch.tensor([5]))
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])


@pytest.mark.parametrize("task", ["style", "tone"])
def test_each_head_backpropagates_through_convolutions_and_embeddings(task):
    model = small_model()
    batch = Collator()([([2, 3, 4, 5, 6, 7], 0, 1), ([8, 9, 10, 11, 12, 13, 14], 2, 3)])
    outputs = model(batch["input_ids"], batch["lengths"])
    F.cross_entropy(outputs[f"{task}_logits"], batch[f"{task}_labels"]).backward()
    assert model.embedding.weight.requires_grad
    gradient = model.embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient[2:].abs().sum() > 0
    assert torch.count_nonzero(gradient[0]) == 0
    for conv in model.convs:
        assert conv.weight.grad is not None
        assert torch.isfinite(conv.weight.grad).all() and conv.weight.grad.abs().sum() > 0
    head = getattr(model, f"{task}_head")
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    other = model.tone_head if task == "style" else model.style_head
    assert all(parameter.grad is None for parameter in other.parameters())


def test_all_padding_backward_is_finite():
    model = small_model()
    batch = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "lengths": torch.zeros(2, dtype=torch.long),
        "style_labels": torch.tensor([0, 1]),
        "tone_labels": torch.tensor([1, 2]),
    }
    loss = multitask_loss(model(batch["input_ids"], batch["lengths"]), batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.embedding.weight.grad) == 0


def test_shared_loader_optimizer_and_evaluation():
    frame = pd.DataFrame(
        {
            "text": [
                "Please send the completed financial report tomorrow.",
                "Hey buddy that was a really great game!",
                "The findings support the proposed experimental research hypothesis.",
            ],
            "style": ["business", "informal", "academic"],
            "tone": ["neutral", "friendly", "neutral"],
        }
    )
    vocab = Vocabulary.fit(frame["text"])
    loader = make_loader(frame, vocab, batch_size=2)
    model = small_model(vocab_size=len(vocab), dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    metrics = run_epoch(model, loader, optimizer=optimizer)
    assert model.training and metrics["loss"] > 0
    for name in ("embedding.weight", "style_head.weight", "tone_head.weight"):
        assert not torch.equal(before[name], dict(model.named_parameters())[name])
    assert torch.count_nonzero(model.embedding.weight[0]) == 0

    optimizer.zero_grad(set_to_none=True)
    evaluated = run_epoch(model, loader)
    batch = next(iter(make_loader(frame, vocab, batch_size=3)))
    with torch.no_grad():
        expected_loss = multitask_loss(model(batch["input_ids"], batch["lengths"]), batch)
    assert not model.training and evaluated["loss"] == pytest.approx(expected_loss.item())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert 0 <= evaluated["style_accuracy"] <= 1
    assert 0 <= evaluated["tone_accuracy"] <= 1


def test_factory_and_checkpoint_roundtrip(tmp_path):
    model = small_model(num_styles=3, num_tones=7).eval()
    assert isinstance(model, TextCNN)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("missing", vocab_size=32)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = small_model(num_styles=3, num_tones=7).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    assert restored(ids)["style_logits"].shape == (1, 3)
    assert restored(ids)["tone_logits"].shape == (1, 7)
    for key, expected in model(ids).items():
        torch.testing.assert_close(expected, restored(ids)[key])