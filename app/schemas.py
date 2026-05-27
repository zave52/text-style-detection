from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class StyleResponse(BaseModel):
    style: str


class ToneResponse(BaseModel):
    tone: str


class PredictResponse(BaseModel):
    style: str
    tone: str
