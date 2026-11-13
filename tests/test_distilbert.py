"""DistilBERT integration tests; all fixtures are small and require no downloads."""

import shutil
import socket
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

from app.config import STYLES, TONES
from app.distilbert import PRETRAINED_NAME
from app.models import build_model
from app.prediction import NeuralPredictor
from app.tokenization import TransformerTokenizer, make_loader
from app.training import model_inputs, multitask_loss, run_epoch


@pytest.fixture
def tokenizer_dir(tmp_path):
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "Hello",
        "hello",
        "world",
        "please",
        "send",
        "report",
        "now",
        "!",
    ]
    backend = Tokenizer(WordPiece({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    backend.normalizer = BertNormalizer(lowercase=False)
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=32,
    )
    path = tmp_path / "base_tokenizer"
    tokenizer.save_pretrained(path)
    return path


def small_model():
    torch.manual_seed(42)
    config = DistilBertConfig(
        vocab_size=13,
        max_position_embeddings=32,
        n_layers=2,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    return build_model("distilbert", encoder_config=config.to_dict(), dropout=0.0)


@pytest.mark.parametrize("limit", [2, 8, 32])
def test_cased_tokenization_limit_and_attention_masks(tokenizer_dir, limit):
    tokenizer = TransformerTokenizer(tokenizer_dir, max_length=limit, local_only=True)
    batch = next(iter(make_loader(["Hello", "please send report now " * 20], tokenizer)))
    assert batch["input_ids"].shape[1] <= limit
    assert torch.equal(batch["attention_mask"], batch["input_ids"].ne(0).long())
    assert torch.equal(batch["attention_mask"].sum(1), batch["lengths"])
    if limit > 2:
        assert tokenizer.encode("Hello") != tokenizer.encode("hello")
    outputs = small_model()(**model_inputs(batch))
    assert outputs["style_logits"].shape == (2, len(STYLES))
    assert outputs["tone_logits"].shape == (2, len(TONES))


def test_pretrained_encoder_is_loaded_for_initial_training(monkeypatch):
    encoder = small_model().encoder
    loader = Mock(return_value=encoder)
    monkeypatch.setattr(DistilBertModel, "from_pretrained", loader)
    model = build_model("distilbert")
    loader.assert_called_once_with(PRETRAINED_NAME, local_files_only=False)
    assert model.encoder is encoder
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


@pytest.mark.parametrize("task", ["style", "tone"])
def test_each_head_backpropagates_into_encoder(tokenizer_dir, task):
    tokenizer = TransformerTokenizer(tokenizer_dir, max_length=8, local_only=True)
    frame = pd.DataFrame(
        {
            "text": ["Hello world", "please send report now!"],
            "style": ["informal", "business"],
            "tone": ["friendly", "urgent"],
        }
    )
    batch = next(iter(make_loader(frame, tokenizer)))
    model = small_model()
    outputs = model(**model_inputs(batch))
    torch.nn.functional.cross_entropy(outputs[f"{task}_logits"], batch[f"{task}_labels"]).backward()
    embedding = model.encoder.embeddings.word_embeddings.weight
    assert embedding.grad is not None and embedding.grad.abs().sum() > 0
    assert torch.count_nonzero(embedding.grad[0]) == 0
    head = getattr(model, f"{task}_head")
    for module in (model.encoder, model.pre_classifier, head):
        for parameter in module.parameters():
            assert parameter.requires_grad and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    assert head.weight.grad.abs().sum() > 0
    other = model.tone_head if task == "style" else model.style_head
    assert all(parameter.grad is None for parameter in other.parameters())


def test_masked_tokens_do_not_change_classification(tokenizer_dir):
    tokenizer = TransformerTokenizer(tokenizer_dir, max_length=16, local_only=True)
    model = small_model().eval()
    batch = next(iter(make_loader(["Hello", "please send report now!"], tokenizer)))
    changed = dict(batch)
    changed["input_ids"] = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, 5)
    with torch.no_grad():
        expected = model(**model_inputs(batch))
        actual = model(**model_inputs(changed))
        alone = model(**model_inputs(next(iter(make_loader(["Hello"], tokenizer)))))
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])
        torch.testing.assert_close(expected[key][:1], alone[key], atol=1e-6, rtol=1e-5)


def test_training_and_relocated_bundle_work_without_network(tokenizer_dir, tmp_path, monkeypatch):
    tokenizer = TransformerTokenizer(tokenizer_dir, max_length=8, local_only=True)
    frame = pd.DataFrame(
        {
            "text": ["Hello world", "please send report now!"],
            "style": ["informal", "business"],
            "tone": ["friendly", "urgent"],
        }
    )
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    metrics = run_epoch(model, make_loader(frame, tokenizer), optimizer=optimizer)
    assert metrics["loss"] > 0
    for name in (
        "encoder.embeddings.word_embeddings.weight",
        "style_head.weight",
        "tone_head.weight",
    ):
        assert not torch.equal(before[name], dict(model.named_parameters())[name])
    model.eval()
    batch = next(iter(make_loader(frame, tokenizer)))
    with torch.no_grad():
        expected = model(**model_inputs(batch))
    original = tmp_path / "original"
    model.save_bundle(original, tokenizer, metadata={"epoch": 1})
    assert (original / "config.json").is_file()
    assert (original / "tokenizer" / "tokenizer.json").is_file()
    copied = tmp_path / "moved_to_another_computer"
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    shutil.rmtree(tokenizer_dir)

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline inference attempted to access the network or base model.")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty_cache"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(DistilBertModel, "from_pretrained", forbidden)
    predictor = NeuralPredictor(copied, device="cpu")
    assert predictor.tokenizer.max_length == 8
    assert predictor.model.encoder.config.to_dict()["dim"] == 16
    with torch.no_grad():
        restored = predictor.model(**model_inputs(batch))
    for key in expected:
        torch.testing.assert_close(expected[key], restored[key])
    assert torch.isfinite(multitask_loss(restored, batch))
    assert predictor.predict(frame["text"].tolist(), batch_size=1) == [
        {"style": STYLES[s], "tone": TONES[t]}
        for s, t in zip(
            expected["style_logits"].argmax(1).tolist(),
            expected["tone_logits"].argmax(1).tolist(),
            strict=True,
        )
    ]
