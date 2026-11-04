"""Shared project configuration.

Single source of truth for the classification scope (language, style and tone
labels), file system paths, random seed, input limits, and runtime device.
The module depends only on the standard library so that it can be imported
cheaply from the API, the Streamlit interface, scripts, notebooks, and tests.
``torch`` is imported lazily and only inside the helpers that need it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# Classification scope
# --------------------------------------------------------------------------- #

LANGUAGE = "en"  # Only English input is supported.
LANGUAGE_NAME = "English"

STYLES: tuple[str, ...] = ("academic", "business", "formal", "informal", "literary")
TONES: tuple[str, ...] = ("aggressive", "friendly", "neutral", "sarcastic", "urgent")

# The raw corpus and the legacy models use the misspelled label "literaly".
# Aliases map every known variant to its canonical label.
LABEL_ALIASES: dict[str, str] = {
    "literaly": "literary",
}

STYLE_TO_ID: dict[str, int] = {label: i for i, label in enumerate(STYLES)}
TONE_TO_ID: dict[str, int] = {label: i for i, label in enumerate(TONES)}
ID_TO_STYLE: dict[int, str] = {i: label for label, i in STYLE_TO_ID.items()}
ID_TO_TONE: dict[int, str] = {i: label for label, i in TONE_TO_ID.items()}


def _normalize(label: str, allowed: tuple[str, ...], kind: str) -> str:
    value = LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())
    if value not in allowed:
        raise ValueError(f"Unknown {kind} label {label!r}. Allowed: {', '.join(allowed)}.")
    return value


def normalize_style(label: str) -> str:
    """Return the canonical style label or raise ``ValueError``."""
    return _normalize(label, STYLES, "style")


def normalize_tone(label: str) -> str:
    """Return the canonical tone label or raise ``ValueError``."""
    return _normalize(label, TONES, "tone")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_DIR = PROJECT_ROOT / "dataset"  # dataset/<style>/<tone>/*.txt
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"

LEGACY_MODELS_DIR = PROJECT_ROOT / "saving"  # TF-IDF + LinearSVC from the ML course
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # neural checkpoints, tokenizers, calibration
REPORTS_DIR = PROJECT_ROOT / "reports"  # metrics, confusion matrices, error analysis

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed ``random``, NumPy, and PyTorch (when installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Input limits (the corpus contains texts of 10-800 characters)
# --------------------------------------------------------------------------- #

MIN_INPUT_CHARS = 1
MAX_INPUT_CHARS = 2000
MAX_BATCH_SIZE = 64

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #

API_HOST = os.getenv("TSD_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("TSD_API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

# "cpu" (default), "cuda", or "auto" (CUDA when available, otherwise CPU).
DEVICE_PREFERENCE = os.getenv("TSD_DEVICE", "cpu").strip().lower()


def resolve_device(preference: str | None = None) -> str:
    """Resolve the PyTorch device name. Defaults to CPU unless requested otherwise."""
    preference = (preference or DEVICE_PREFERENCE).strip().lower()
    if preference == "cpu":
        return "cpu"
    if preference not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device {preference!r}. Use 'cpu', 'cuda', or 'auto'.")

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError("TSD_DEVICE=cuda was requested but CUDA is not available.")
    return "cpu"
