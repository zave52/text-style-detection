from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.config import MAX_BATCH_SIZE, MAX_INPUT_CHARS, MIN_INPUT_CHARS
from app.input_validation import validate_batch, validate_text


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=MIN_INPUT_CHARS, max_length=MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_text(value)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[StrictStr] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def valid_texts(cls, value: list[str]) -> list[str]:
        return validate_batch(value)


class PredictionWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal["truncated", "unfamiliar_input", "uncertain_prediction"]
    task: Literal["style", "tone"] | None
    message: str


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StyleResponse(BaseModel):
    style: str


class ToneResponse(BaseModel):
    tone: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
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
    results: list[PredictResponse]
