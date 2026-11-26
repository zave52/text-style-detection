"""FastAPI adapter for the shared calibrated prediction service."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.input_validation import PredictionInputError
from app.schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    LivenessResponse,
    ModelInfoResponse,
    PredictResponse,
    ReadinessResponse,
    TextRequest,
)
from app.service import PredictionService, get_prediction_service

logger = logging.getLogger(__name__)

MODEL_UNAVAILABLE = "Prediction model unavailable. Select a calibrated model and restart the API."
PREDICTION_FAILED = "Prediction could not be completed."
BATCH_PREDICTION_FAILED = "Batch prediction could not be completed."

DESCRIPTION = """
Classify the **writing style** and **emotional tone** of English messages with the
selected calibrated neural model (TextCNN, BiLSTM with attention, or DistilBERT).

* **Prediction** endpoints return labels, calibrated confidence, every class score,
  token information, and warnings.
* **Health** endpoints separate process liveness from model readiness: the process
  stays live when the model fails to load, while readiness and prediction return `503`.
"""

TAGS = [
    {"name": "Prediction", "description": "Style and tone classification."},
    {"name": "Model", "description": "Metadata of the loaded model."},
    {"name": "Health", "description": "Liveness and readiness probes."},
]

UNAVAILABLE = {
    503: {"model": ErrorResponse, "description": "Model not loaded or prediction failed"}
}
INVALID = {422: {"description": "Invalid request body"}}


def create_app(service: PredictionService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.prediction_service = service
        if service is None:
            try:
                application.state.prediction_service = get_prediction_service()
            except Exception:
                logger.exception("Selected prediction model could not be loaded")
        try:
            yield
        finally:
            application.state.prediction_service = None
            if service is None:
                get_prediction_service.cache_clear()

    application = FastAPI(
        title="Text Style & Tone Detection API",
        description=DESCRIPTION,
        version="0.2.0",
        openapi_tags=TAGS,
        lifespan=lifespan,
    )

    def ready_service() -> PredictionService:
        loaded = getattr(application.state, "prediction_service", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)
        return loaded

    Service = Annotated[PredictionService, Depends(ready_service)]

    def run(predict, payload, failure: str):
        try:
            return predict(payload)
        except PredictionInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logger.exception(failure)
            raise HTTPException(status_code=503, detail=failure) from error

    @application.get(
        "/health/live",
        response_model=LivenessResponse,
        tags=["Health"],
        summary="Process liveness",
    )
    def live():
        """Report that the API process is serving requests, even without a model."""
        return {"status": "alive"}

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses=UNAVAILABLE,
        tags=["Health"],
        summary="Model readiness",
    )
    def ready(predictor: Service):
        """Report whether a verified, calibrated model is loaded and can serve predictions."""
        return {"status": "ready", "model": predictor.info()["model"]}

    @application.get(
        "/model/info",
        response_model=ModelInfoResponse,
        responses=UNAVAILABLE,
        tags=["Model"],
        summary="Loaded model metadata",
    )
    def model_info(predictor: Service):
        """Return architecture, labels, token limit, temperatures, input limits and thresholds."""
        return predictor.info()

    def predict_route(path: str, summary: str, description: str):
        @application.post(
            path,
            response_model=PredictResponse,
            responses={**INVALID, **UNAVAILABLE},
            tags=["Prediction"],
            summary=summary,
            description=description,
        )
        def predict(request: TextRequest, predictor: Service):
            return run(predictor.predict, request.text, PREDICTION_FAILED)

        return predict

    full = "Both tasks and every prediction field are returned."
    predict_route("/predict", "Predict style and tone", f"Classify one message. {full}")
    predict_route(
        "/predict/style",
        "Predict style",
        f"Style-focused alias of `/predict` used by the UI. {full}",
    )
    predict_route(
        "/predict/tone",
        "Predict tone",
        f"Tone-focused alias of `/predict` used by the UI. {full}",
    )

    @application.post(
        "/predict/batch",
        response_model=BatchResponse,
        responses={**INVALID, **UNAVAILABLE},
        tags=["Prediction"],
        summary="Predict a batch of messages",
    )
    def predict_batch(request: BatchRequest, predictor: Service):
        """Classify up to the configured number of messages; results keep the input order."""
        return {"results": run(predictor.predict_batch, request.texts, BATCH_PREDICTION_FAILED)}

    return application


app = create_app()
