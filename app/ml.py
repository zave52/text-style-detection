from pathlib import Path

import joblib
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent
STYLE_MODEL_PATH = BASE_DIR / "saving" / "style_model.pkl"
TONE_MODEL_PATH = BASE_DIR / "saving" / "tone_model.pkl"

models = {}


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
    return str(models["style"].predict([text])[0])


def predict_tone(text: str) -> str:
    check_models_loaded(["tone"])
    return str(models["tone"].predict([text])[0])


def predict_both(text: str) -> tuple[str, str]:
    check_models_loaded(["style", "tone"])
    style_pred = str(models["style"].predict([text])[0])
    tone_pred = str(models["tone"].predict([text])[0])
    return style_pred, tone_pred
