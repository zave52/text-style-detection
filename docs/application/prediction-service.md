# Shared Prediction Service

`app.service.PredictionService` provides the same calibrated prediction contract to Python callers, the CLI, FastAPI, and Streamlit. The API uses the selected neural checkpoint instead of the legacy TF-IDF models.

## Select a Model

Select a complete, calibrated shared-pipeline bundle:

```bash
uv run python -m scripts.select_model --checkpoint artifacts/experiments/distilbert_01
```

The service verifies the registry's pinned bundle checksum, the bundle file checksums, checkpoint metadata, label order, tokenizer compatibility, sequence limit, and calibration parameters. Transformer configuration, tokenizer files, and weights are loaded locally. No base-model download is needed for inference.

The service is loaded once per API process. After selecting another checkpoint, restart the API. Python callers can construct a new service instance. `TSD_DEVICE` controls the default device; `TSD_CHECKPOINT` can explicitly select a bundle and `TSD_REGISTRY` can select a registry file.

## Python

```python
from app.service import PredictionService

service = PredictionService()
single = service.predict("Please send the financial report tomorrow.")
batch = service.predict_batch(["Hello!", "Please send the financial report tomorrow."])
metadata = service.info()
```

An explicit local bundle can also be used:

```python
service = PredictionService("artifacts/experiments/bilstm_01", device="cpu")
```

The lower-level `NeuralPredictor.predict` and `predict_proba` methods remain available for existing callers, evaluation, and calibration. The public `PredictionService` additionally requires a verified calibrated bundle, validates request limits, and returns the full response.

## CLI

```bash
uv run python -m app.prediction "Please send the financial report tomorrow."
uv run python -m app.prediction "Hello!" "Please send the financial report tomorrow."
```

One message returns a prediction object. Multiple messages return `{"results": [...]}`. Every prediction object has the same fields as the API response. `--checkpoint`, `--registry`, `--device`, and `--batch-size` are supported. The previous `--probabilities` flag remains accepted; class scores are now always included.

## API and Interface

Start the API and UI in separate terminals:

```bash
uv run uvicorn app.main:app --reload
uv run streamlit run frontend/app.py
```

| Endpoint | Input or purpose | Response |
|---|---|---|
| `POST /predict` | `{"text": "Hello!"}` | Full prediction object |
| `POST /predict/batch` | `{"texts": ["Hello!", "Please send the report."]}` | `{"results": [...]}` |
| `POST /predict/style` | Compatibility path for a single message | Full prediction object |
| `POST /predict/tone` | Compatibility path for a single message | Full prediction object |
| `GET /model/info` | Loaded model, labels, checksums, limits, temperatures, and warning thresholds | Model information |
| `GET /health/live` | Process liveness | `200` while the application is running |
| `GET /health/ready` | Model availability | `200` when a model is loaded; otherwise `503` |

The compatibility paths return the full shared response; the interface displays only the requested task. Invalid request bodies return `422`. An unavailable model or failed inference returns `503`; loading failures are logged on the server without exposing filesystem paths in the response.

The Streamlit single-message view displays labels, confidence, class scores, token counts, and warnings. The batch view accepts one message per line, ignoring blank lines, and displays one result per submitted message. Both views consume API responses and can display the complete JSON result. The frontend reads `API_URL` from the environment and runs with only the `ui` dependency group; it does not import `app.*` or duplicate input validation. Invalid requests are sent to the API and its `422` details are displayed.

For Compose deployment, keep the selected registry and complete bundle under host `artifacts/`. The backend mounts this tree read-only and reports model readiness through `/health/ready`; its dependents wait for the image's healthcheck. See the [deployment guide](deployment.md).

## Prediction Fields

| Field | Meaning |
|---|---|
| `model` | Loaded architecture: `textcnn`, `bilstm`, or `distilbert` |
| `style`, `tone` | Predicted labels |
| `style_confidence`, `tone_confidence` | Maximum calibrated probability for each task |
| `style_scores`, `tone_scores` | Probabilities for all five classes in saved label order |
| `character_count` | Original message length in characters |
| `token_count` | Token count before truncation |
| `used_token_count` | Number of encoded tokens retained for inference, excluding batch padding |
| `max_length` | Saved sequence-length limit |
| `truncated` | Whether the original token count exceeded the limit |
| `unknown_token_count` | Unknown tokens in the retained content |
| `unknown_ratio` | Unknown-token fraction among retained content tokens |
| `warnings` | Structured objects containing `code`, `task`, and `message` |

For DistilBERT, token counts include the tokenizer's special tokens. The unknown ratio excludes special tokens other than UNK, so `[UNK]` remains counted as unknown. Counts describe the saved tokenizer; they are not word counts. Input text is validated without stripping or otherwise rewriting the message sent to the model.

## Validation and Warnings

The public contract accepts non-empty, non-whitespace strings of **1–2000 characters** and batches of **1–64 messages**. Internal inference batch size must also be an integer in `1–64`. Limits come from `app.config`. The complete batch is validated before tokenization and inference; results preserve input order, including duplicate messages.

| Warning code | Default trigger |
|---|---|
| `truncated` | Token count exceeds the saved sequence limit |
| `unfamiliar_input` | Retained unknown ratio is greater than `0.30`, there are no alphabetic characters, or most letters are non-Latin |
| `uncertain_prediction` | Per-task maximum probability is below `0.55`, or the gap between the two highest probabilities is below `0.15` |

General warnings have `task: null`; uncertainty warnings identify `style` or `tone`. Thresholds can be configured when constructing `PredictionService` and are returned by `info()`.

Unfamiliar-input checks are heuristics, not a language detector or a complete out-of-distribution test. Confidence values are model estimates, not guarantees of correctness. Warnings do not change labels, class scores, or saved calibration temperatures.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Regression tests cover all three architectures, offline loading from read-only artifacts, checksum failures, saved temperature application, exact single/batch/CLI/API result parity, request limits, warnings, process caching, readiness failures, API validation errors in Streamlit, and standalone frontend startup without backend modules.
