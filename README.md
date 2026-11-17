# Text Style & Tone Detection

Classify the writing style and emotional tone of English messages with **TextCNN, BiLSTM with attention, or DistilBERT**. A shared prediction service loads a selected, calibrated neural checkpoint and provides the same response through Python, the CLI, FastAPI, and Streamlit.

The original TF-IDF and spaCy classifiers remain documented as historical baselines. The application serves the selected neural architecture; it does not load `saving/*.joblib` or run spaCy preprocessing during prediction.

## Classification scope

| Property | Values |
|---|---|
| Styles | `academic`, `business`, `formal`, `informal`, `literary` |
| Tones | `aggressive`, `friendly`, `neutral`, `sarcastic`, `urgent` |
| Language | English |
| Request limits | 1–2000 characters per non-whitespace message; 1–64 messages per batch |

The API owns input validation. Streamlit sends requests over HTTP and displays API validation errors, predictions, calibrated confidence, class scores, token information, and warnings. Its image requires only the `ui` dependency group and the `frontend/` files.

## Models and experiments

| Architecture | Representation |
|---|---|
| TextCNN | Trainable embeddings, convolution windows of 3/4/5 tokens, padding-safe pooling |
| BiLSTM | Trainable embeddings, packed bidirectional sequences, masked attention and final hidden states |
| DistilBERT | Fine-tuned English cased encoder with its saved tokenizer and attention masks |

Each architecture has separate style and tone heads. The shared training pipeline combines the task losses, uses AdamW, clipping and scheduling, and selects checkpoints by **validation mean macro F1** with early stopping. Separate temperatures are fitted on validation predictions. Test data is excluded from model selection and calibration fitting.

Bundles preserve model weights, labels, tokenizer or vocabulary, sequence limit, calibration, configuration, training metadata, history, and dataset manifest. Loading checks file and weight checksums before inference. DistilBERT inference uses the saved configuration and local tokenizer without downloading its base model.

The raw corpus has 1000 messages. The audited dataset excludes two conflicting records and contains **998 messages**, split into **632 training / 158 validation / 208 test** examples with group-aware leakage checks. See [dataset preparation](docs/dataset/preprocessing.md).

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root. Python 3.13 and CPU PyTorch are configured in `pyproject.toml` and `uv.lock`.

```bash
uv sync --frozen
```

Select an existing **complete, calibrated bundle**, keeping its entire experiment directory:

```bash
uv run python -m scripts.select_model --checkpoint artifacts/experiments/distilbert_01
```

The command validates the bundle and records it in `artifacts/active_model.json`. If no bundle exists yet, prepare data and train one first, using an unused run directory:

```bash
uv run python -m scripts.build_dataset
uv run python -m scripts.train --model textcnn --run-dir artifacts/experiments/textcnn_01 --epochs 20
uv run python -m scripts.select_model --checkpoint artifacts/experiments/textcnn_01
```

Start the API and UI in **separate terminals**:

```bash
uv run uvicorn app.main:app --reload
```

```bash
uv run streamlit run frontend/app.py
```

Open the UI at `http://localhost:8501`, Swagger at `http://localhost:8000/docs`, and model information at `http://localhost:8000/model/info`.

The CLI uses the same selected model:

```bash
uv run python -m app.prediction "Please send the financial report tomorrow."
uv run python -m app.prediction --batch-size 4 "Hello!" "Please send the report."
```

## Docker Compose

Prepare and select a bundle on the host as above **before starting Compose**. Keep the selected bundle inside `artifacts/`; registry paths are stored relative to the registry and remain usable at the container mount point.

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f backend
```

Compose mounts `./artifacts` at `/app/artifacts` **read-only**. Weights are excluded from the image build context. The backend image checks `http://127.0.0.1:8000/health/ready`; Compose waits for the backend to become healthy before starting the UI and proxy. An absent or invalid model makes readiness fail and is reported in backend logs.

Nginx exposes `http://localhost:8000` with the UI at `/`, Swagger at `/docs`, prediction paths at `/predict...`, health at `/health/...`, and model metadata at `/model/info`. Backend and frontend ports remain internal to Compose.

After selecting another model, reload the running API: restart Uvicorn locally or run `docker compose restart backend`. Selection does not replace a model already loaded into memory. The backend container reads the registry but cannot modify the mounted artifacts.

See the [deployment guide](docs/application/deployment.md) for model preparation, configuration, and troubleshooting.

## Prediction API

| Method and path | Purpose |
|---|---|
| `POST /predict` | One message: `{"text": "Hello!"}` |
| `POST /predict/batch` | Ordered batch: `{"texts": ["Hello!", "Please send the report."]}` |
| `POST /predict/style`, `POST /predict/tone` | Compatibility paths returning the full two-task response |
| `GET /model/info` | Model, labels, tokenizer limit, temperatures, checksums, and input limits |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | `200` when the model is loaded; `503` otherwise |

A single prediction has this shape; the values below are illustrative, not benchmark results:

```json
{
  "model": "distilbert",
  "style": "informal",
  "tone": "friendly",
  "style_confidence": 0.83,
  "tone_confidence": 0.8,
  "style_scores": {"academic": 0.03, "business": 0.05, "formal": 0.04, "informal": 0.83, "literary": 0.05},
  "tone_scores": {"aggressive": 0.02, "friendly": 0.8, "neutral": 0.08, "sarcastic": 0.05, "urgent": 0.05},
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

Batch responses wrap the same objects in `{"results": [...]}`. Scores are normalized independently for the two tasks. Warnings identify truncation, unfamiliar input, or uncertain predictions. Unfamiliar-input warnings are heuristics rather than a language detector.

Invalid bodies, whitespace-only input, oversized messages, and invalid batch sizes return `422`. Model loading or inference failures return `503`. See the [API reference](docs/application/api.md) and [shared prediction service](docs/application/prediction-service.md) for the full contract.

## Repository guide

| Path | Responsibility |
|---|---|
| `app/models.py`, `app/bilstm.py`, `app/distilbert.py` | Neural architectures |
| `app/tokenization.py` | Saved vocabulary and transformer tokenizer interfaces |
| `app/experiments.py`, `app/training.py` | Shared training and checkpoint selection |
| `app/calibration.py`, `app/bundles.py`, `app/registry.py` | Calibration, integrity, and active-model selection |
| `app/service.py`, `app/prediction.py` | Reusable prediction service and CLI |
| `app/main.py`, `app/schemas.py`, `app/input_validation.py` | API lifecycle, schemas, and validation |
| `frontend/app.py` | Independent Streamlit HTTP client |
| `scripts/train.py`, `scripts/calibrate.py`, `scripts/select_model.py` | Experiment and selection commands |
| `scripts/evaluate.py` | Compatible held-out evaluation and exported reports |
| `artifacts/` | Local trained bundles and active registry, mounted for inference |
| `dataset/`, `data/prepared/`, `reports/` | Raw corpus, prepared splits, and analysis |
| `app/ml.py`, analysis notebooks | Historical classical models |

## Verification and documentation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run --group docs mkdocs build --strict
```

The [architecture guide](docs/application/architecture.md) describes service boundaries and model loading. [Classical vs neural](docs/modeling/legacy-vs-neural.md) reports historical accuracy and F1 with split and averaging caveats; those results are not a benchmark of whichever bundle is currently selected.

Documentation is configured in `mkdocs.yml` and published at [zave52.github.io/text-style-detection](https://zave52.github.io/text-style-detection/).
