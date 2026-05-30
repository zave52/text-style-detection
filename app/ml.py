import re
from pathlib import Path

import joblib
import spacy
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent
STYLE_MODEL_PATH = BASE_DIR / "saving" / "style_model.pkl"
TONE_MODEL_PATH = BASE_DIR / "saving" / "tone_model.pkl"

models = {}

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli

    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    doc = nlp(text)
    clean_tokens = []

    for token in doc:
        if token.is_space:
            continue
        if token.is_punct:
            clean_tokens.append(token.text)
        else:
            lemma = token.lemma_.strip()
            if lemma:
                clean_tokens.append(lemma.lower())

    clean_text = " ".join(clean_tokens)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text


def load_models():
    try:
        if STYLE_MODEL_PATH.exists():
            models["style"] = joblib.load(STYLE_MODEL_PATH)
            print("Style model loaded successfully.")
        else:
            print(f"Warning: {STYLE_MODEL_PATH} not found.")

        if TONE_MODEL_PATH.exists():
            models["tone"] = joblib.load(TONE_MODEL_PATH)
            print("Tone model loaded successfully.")
        else:
            print(f"Warning: {TONE_MODEL_PATH} not found.")

    except Exception as e:
        print(f"Error loading models: {e}")


def clear_models():
    models.clear()


def check_models_loaded(model_keys):
    for key in model_keys:
        if key not in models:
            raise HTTPException(
                status_code=503,
                detail=f"{key.capitalize()} model is not loaded."
            )


def predict_style(text: str) -> str:
    check_models_loaded(["style"])
    clean_text = preprocess_text(text)
    return str(models["style"].predict([clean_text])[0])


def predict_tone(text: str) -> str:
    check_models_loaded(["tone"])
    clean_text = preprocess_text(text)
    return str(models["tone"].predict([clean_text])[0])


def predict_both(text: str) -> tuple[str, str]:
    check_models_loaded(["style", "tone"])
    clean_text = preprocess_text(text)
    style_pred = str(models["style"].predict([clean_text])[0])
    tone_pred = str(models["tone"].predict([clean_text])[0])
    return style_pred, tone_pred
