"""Request and response contracts of the public prediction API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.config import MAX_BATCH_SIZE, MAX_INPUT_CHARS, MIN_INPUT_CHARS
from app.input_validation import validate_batch, validate_text

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
ModelName = Literal["textcnn", "bilstm", "distilbert"]


class TextRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"text": "Please send the financial report tomorrow."}]},
    )
    text: StrictStr = Field(
        min_length=MIN_INPUT_CHARS,
        max_length=MAX_INPUT_CHARS,
        description="English message; must not be empty or whitespace-only.",
    )

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_text(value)


class BatchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"texts": ["Hello! How are you?", "Submit the report by noon."]}]
        },
    )
    texts: list[StrictStr] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Messages to classify; results keep the same order.",
    )

    @field_validator("texts")
    @classmethod
    def valid_texts(cls, value: list[str]) -> list[str]:
        return validate_batch(value)


class PredictionWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal["truncated", "unfamiliar_input", "uncertain_prediction"]
    task: Literal["style", "tone"] | None
    message: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelName
    style: str
    tone: str
    style_confidence: Probability
    tone_confidence: Probability
    style_scores: dict[str, Probability]
    tone_scores: dict[str, Probability]
    character_count: int = Field(ge=1)
    token_count: int = Field(ge=0)
    used_token_count: int = Field(ge=1)
    max_length: int = Field(ge=1)
    truncated: bool
    unknown_token_count: int = Field(ge=0)
    unknown_ratio: Probability
    warnings: list[PredictionWarning]


class BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[PredictResponse]


class InputLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_characters: int = Field(ge=1)
    max_characters: int = Field(ge=1)
    max_batch_size: int = Field(ge=1)


class WarningThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence: Probability
    margin: Probability
    unknown_ratio: Probability


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelName
    language: str
    styles: list[str]
    tones: list[str]
    max_length: int = Field(ge=1)
    tokenizer_kind: Literal["vocabulary", "transformer"]
    device: str
    temperatures: dict[Literal["style", "tone"], float]
    weights_sha256: str
    input_limits: InputLimits
    warning_thresholds: WarningThresholds


class LivenessResponse(BaseModel):
    status: Literal["alive"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    model: ModelName


class ErrorResponse(BaseModel):
    detail: str
