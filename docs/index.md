---
hide:
  - navigation
---

# Text Style & Tone Detection

Classify English writing style and emotional tone with a selected **TextCNN, BiLSTM with attention, or DistilBERT** checkpoint. The CLI, Python interface, and FastAPI backend use one calibrated prediction service. Streamlit is a separate HTTP client.

## Classification scope

| Property | Classes |
|---|---|
| Style | `academic`, `business`, `formal`, `informal`, `literary` |
| Tone | `aggressive`, `friendly`, `neutral`, `sarcastic`, `urgent` |

The API accepts non-whitespace messages of 1–2000 characters and batches of 1–64 messages. Predictions contain both labels, calibrated confidence and class scores, token statistics, and structured warnings for truncation, unfamiliar input, and uncertainty.

## Training and evaluation

The shared pipeline provides AdamW, gradient clipping, scheduling, early stopping, and selection by validation mean macro F1. Temperatures for style and tone are fitted on validation predictions only. Prepared data contains 998 audited messages with a 632/158/208 train/validation/test split and group-aware leakage checks.

Saved bundles preserve weights, tokenizer or vocabulary, labels, configuration, calibration, training metadata, and dataset fingerprints. File checksums and registry pins are checked during loading. DistilBERT inference is local and does not download a base model.

Historical TF-IDF and spaCy models remain available in the notebooks. Their reported results use different evaluation protocols; see [Classical vs Neural](modeling/legacy-vs-neural.md) for the comparison and limitations. They are not used by the current API.

## Quick start

Install uv and use the configured Python 3.13 environment:

```bash
uv sync --frozen
uv run python -m scripts.select_model --checkpoint artifacts/experiments/distilbert_01
```

The selection command requires an existing complete calibrated bundle. If you do not have one yet, follow [model preparation](application/deployment.md#prepare-and-select-a-model) first.

Start the API and UI in separate terminals:

```bash
uv run uvicorn app.main:app --reload
```

```bash
uv run streamlit run frontend/app.py
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for Swagger.

For Docker, prepare the selected bundle under host `artifacts/`, then run `docker compose up --build -d`. Compose mounts the artifacts read-only and waits for the backend's model readiness check. The full stack is available at `http://localhost:8000`.

## Guides

| Guide | Contents |
|---|---|
| [Prediction Service](application/prediction-service.md) | Python, CLI, calibration, token information, and warnings |
| [API Reference](application/api.md) | Single/batch requests, full response schemas, validation, and health |
| [Architecture](application/architecture.md) | Runtime boundaries, loading, and proxy routes |
| [Deployment](application/deployment.md) | Host model preparation, local startup, read-only mounts, and troubleshooting |
| [Classical vs Neural](modeling/legacy-vs-neural.md) | Historical metrics and fair-comparison requirements |
