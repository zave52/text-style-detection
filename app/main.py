"""FastAPI adapter for the shared calibrated prediction service."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.input_validation import PredictionInputError
from app.schemas import BatchRequest, BatchResponse, PredictResponse, TextRequest
from app.service import PredictionService, get_prediction_service

logger = logging.getLogger(__name__)


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

    application = FastAPI(title="Text Style & Tone Detection API", lifespan=lifespan)

    def ready_service() -> PredictionService:
        loaded = getattr(application.state, "prediction_service", None)
        if loaded is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Prediction model unavailable. Select a calibrated model and restart the API."
                ),
            )
        return loaded

    Service = Annotated[PredictionService, Depends(ready_service)]

    @application.get("/health/live")
    def live():
        return {"status": "alive"}

    @application.get("/health/ready")
    def ready(predictor: Service):
        return {"status": "ready", "model": predictor.info()["model"]}

    @application.get("/model/info")
    def model_info(predictor: Service):
        return predictor.info()

    @application.post("/predict", response_model=PredictResponse)
    @application.post("/predict/style", response_model=PredictResponse)
    @application.post("/predict/tone", response_model=PredictResponse)
    def predict(request: TextRequest, predictor: Service):
        try:
            return predictor.predict(request.text)
        except PredictionInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logger.exception("Prediction failed")
            raise HTTPException(
                status_code=503, detail="Prediction could not be completed."
            ) from error

    @application.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest, predictor: Service):
        try:
            return {"results": predictor.predict_batch(request.texts)}
        except PredictionInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logger.exception("Batch prediction failed")
            raise HTTPException(
                status_code=503, detail="Prediction could not be completed."
            ) from error

    return application


app = create_app()
