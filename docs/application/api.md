# API Reference

FastAPI serves the selected **calibrated neural model** through `app.service.PredictionService`. Input validation and the prediction contract are shared with the CLI and Python callers. Streamlit consumes this API over HTTP.

Locally, the API is at `http://localhost:8000`. Compose exposes the same API paths through Nginx at that address. Interactive documentation is at `/docs`; the generated schema is at `/openapi.json`.

## Endpoints

| Method | Path | Input | Response |
|---|---|---|---|
| POST | `/predict` | `TextRequest` | Full `PredictResponse` |
| POST | `/predict/batch` | `BatchRequest` | `{"results": [PredictResponse, ...]}` |
| POST | `/predict/style` | `TextRequest` | Full `PredictResponse` |
| POST | `/predict/tone` | `TextRequest` | Full `PredictResponse` |
| GET | `/model/info` | None | Loaded model metadata |
| GET | `/health/live` | None | `{"status": "alive"}` |
| GET | `/health/ready` | None | `{"status": "ready", "model": "..."}` or `503` |

The `/predict/style` and `/predict/tone` paths are compatibility aliases. **Both return both tasks and every prediction field.** The UI decides which task to display. They do not expose separate model loaders or a reduced schema.

## Single-message prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!"}'
```

Illustrative response; probabilities depend on the selected checkpoint:

```json
{
  "model": "distilbert",
  "style": "informal",
  "tone": "friendly",
  "style_confidence": 0.83,
  "tone_confidence": 0.8,
  "style_scores": {
    "academic": 0.03,
    "business": 0.05,
    "formal": 0.04,
    "informal": 0.83,
    "literary": 0.05
  },
  "tone_scores": {
    "aggressive": 0.02,
    "friendly": 0.8,
    "neutral": 0.08,
    "sarcastic": 0.05,
    "urgent": 0.05
  },
  "character_count": 6,
  "token_count": 4,
  "used_token_count": 4,
  "max_length": 128,
  "truncated": false,
  "unknown_token_count": 0,
  "unknown_ratio": 0.0,
  "warnings": []
}
```

## Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello!", "Please send the report."]}'
```

The response is `{"results": [...]}`, with one complete prediction object per message in the original order. Duplicates are preserved. The entire batch must pass validation before tokenization and inference.

## Requests and validation

| Schema | Field | Rules |
|---|---|---|
| `TextRequest` | `text` | Required string, 1–2000 characters, not whitespace-only |
| `BatchRequest` | `texts` | Required list of 1–64 strings; every string follows the same text rules |

Non-string values are rejected rather than coerced. Unknown request fields are forbidden. Validation preserves the supplied text, including capitalization, punctuation, and whitespace. The model's tokenizer applies its saved sequence-length limit after character validation.

The canonical limits live in `app/config.py` and are exposed by `GET /model/info`. The frontend does not import backend modules or duplicate text and batch validation; it displays errors returned by the API. In its batch view, blank lines are ignored before forming the list of messages.

## Prediction fields

| Field | Type | Meaning |
|---|---|---|
| `model` | string | `textcnn`, `bilstm`, or `distilbert` |
| `style`, `tone` | string | Highest-probability labels for each task |
| `style_confidence`, `tone_confidence` | number | The corresponding calibrated maximum probability |
| `style_scores`, `tone_scores` | object | All five class probabilities; each task sums to one |
| `character_count` | integer | Original message length |
| `token_count` | integer | Saved tokenizer's token count before truncation |
| `used_token_count` | integer | Retained encoded tokens, excluding batch padding |
| `max_length` | integer | Sequence limit saved with the model |
| `truncated` | boolean | Whether the pre-truncation token count exceeded that limit |
| `unknown_token_count` | integer | Unknown tokens among retained content tokens |
| `unknown_ratio` | number | Their fraction among retained content tokens |
| `warnings` | array | Objects containing `code`, `task`, and `message` |

DistilBERT token counts include special tokens such as `[CLS]` and `[SEP]`. The unknown ratio excludes special tokens except `[UNK]`. Token counts are tokenizer-specific, not word counts.

| Task | Canonical labels |
|---|---|
| Style | `academic`, `business`, `formal`, `informal`, `literary` |
| Tone | `aggressive`, `friendly`, `neutral`, `sarcastic`, `urgent` |

## Warnings

| Code | Default condition | `task` |
|---|---|---|
| `truncated` | Original token count exceeds the saved limit | `null` |
| `unfamiliar_input` | Unknown ratio > 0.30, no alphabetic characters, or predominantly non-Latin letters | `null` |
| `uncertain_prediction` | Maximum probability < 0.55 or top-two probability gap < 0.15 | `style` or `tone` |

For example:

```json
{
  "code": "uncertain_prediction",
  "task": "style",
  "message": "The style prediction is uncertain; inspect the class scores."
}
```

Warnings are informational and do not change predictions or temperatures. Unfamiliar-input checks are heuristics rather than a language detector. Thresholds used by the running service are returned by `/model/info`.

## Model information and health

`GET /model/info` returns the architecture, English language scope, labels, saved token limit, tokenizer kind, device, style and tone temperatures, weight SHA-256, input limits, and warning thresholds. It does not expose local filesystem paths.

`GET /health/live` confirms that the process is serving requests. `GET /health/ready` requires a successfully loaded and verified model. A model loading failure is logged during startup; the process remains live while readiness and prediction endpoints return `503`. Loading is attempted once per API process, so fix the artifacts or selection and restart the backend.

Compose uses readiness for its backend healthcheck. The registry and model bundle are mounted read-only; changing the active selection is a host-side operation followed by an API restart.

## Errors

### 422 — Invalid request

Missing fields, wrong types, whitespace-only input, oversized messages, and empty or oversized batches are rejected. Schema errors use FastAPI's `detail` list; errors from shared validation may use a `detail` string. Exact validation wording can vary with the installed Pydantic version.

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### 503 — Model unavailable

```json
{
  "detail": "Prediction model unavailable. Select a calibrated model and restart the API."
}
```

### 503 — Inference failure

```json
{
  "detail": "Prediction could not be completed."
}
```

The batch equivalent says `Batch prediction could not be completed.` Internal loading and inference exceptions are recorded in backend logs rather than exposed in the response. See the [deployment guide](deployment.md) for readiness failures and [prediction service](prediction-service.md) for Python and CLI use.
