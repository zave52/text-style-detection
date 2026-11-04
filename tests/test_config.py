import pytest

from app import config


def test_scope_is_five_styles_and_five_tones():
    assert len(config.STYLES) == 5
    assert len(config.TONES) == 5
    assert len(set(config.STYLES)) == 5
    assert len(set(config.TONES)) == 5
    assert config.LANGUAGE == "en"


def test_label_id_maps_are_consistent():
    for i, label in enumerate(config.STYLES):
        assert config.STYLE_TO_ID[label] == i
        assert config.ID_TO_STYLE[i] == label
    for i, label in enumerate(config.TONES):
        assert config.TONE_TO_ID[label] == i
        assert config.ID_TO_TONE[i] == label


def test_aliases_point_to_canonical_labels():
    assert set(config.LABEL_ALIASES.values()) <= set(config.STYLES) | set(config.TONES)
    assert config.normalize_style("literaly") == "literary"
    assert config.normalize_style("  Formal ") == "formal"
    assert config.normalize_tone("Sarcastic") == "sarcastic"


@pytest.mark.parametrize("func", [config.normalize_style, config.normalize_tone])
def test_unknown_labels_are_rejected(func):
    with pytest.raises(ValueError):
        func("unknown")


def test_paths_are_anchored_at_project_root():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert config.DATASET_CSV.parent == config.DATA_DIR
    assert config.RAW_DATASET_DIR.parent == config.PROJECT_ROOT


def test_input_limits_are_sane():
    assert 0 < config.MIN_INPUT_CHARS < config.MAX_INPUT_CHARS
    assert config.MAX_BATCH_SIZE > 0


def test_cpu_is_default_device(monkeypatch):
    assert config.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        config.resolve_device("tpu")


def test_set_seed_is_reproducible():
    import random

    config.set_seed(123)
    first = random.random()
    config.set_seed(123)
    assert random.random() == first
