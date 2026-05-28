from contextlib import asynccontextmanager

from fastapi import FastAPI

from ml import (
    load_models,
    clear_models,
    predict_style,
    predict_tone,
    predict_both
)
from schemas import (
    TextRequest,
    StyleResponse,
    ToneResponse,
    PredictResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    clear_models()


app = FastAPI(title="Text Style & Tone Detection API", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
async def predict_both_endpoint(request: TextRequest):
    style_pred, tone_pred = predict_both(request.text)
    return PredictResponse(style=style_pred, tone=tone_pred)


@app.post("/predict/style", response_model=StyleResponse)
async def predict_style_endpoint(request: TextRequest):
    style_pred = predict_style(request.text)
    return StyleResponse(style=style_pred)


@app.post("/predict/tone", response_model=ToneResponse)
async def predict_tone_endpoint(request: TextRequest):
    tone_pred = predict_tone(request.text)
    return ToneResponse(tone=tone_pred)
