# Launch Modes and Model Selection

This document details all execution modes for the Text Style & Tone Detection system: the command-line interface (CLI), the FastAPI backend, the Streamlit frontend, and the multi-container Docker Compose deployment. It also explains how to select and switch active model checkpoints.

---

## 1. Model Selection Workflow

The application serves a single active neural model selected from a complete, calibrated experiment bundle. The registry file `artifacts/active_model.json` records the active bundle path, architecture, and SHA-256 checksums.

### Inspecting Registered Models

```bash
uv run python -m scripts.select_model --list
```

### Selecting a Checkpoint

Point `--checkpoint` to an experiment directory (or directly to its `best.pt`):

```bash
uv run python -m scripts.select_model --checkpoint artifacts/experiments/distilbert_01
```

Or for other architectures:
```bash
uv run python -m scripts.select_model --checkpoint artifacts/experiments/textcnn_01
uv run python -m scripts.select_model --checkpoint artifacts/experiments/bilstm_01
```

### Bundle Integrity Verification

During selection, the script checks:
- Required files: `bundle.json`, `best.pt`, `calibration.json`, `labels.json`, `model_config.json`, `training_metadata.json`, `training_config.json`, `data_manifest.json`, `history.json`, `summary.json`.
- Word models (`textcnn`, `bilstm`) must contain `vocabulary.json`.
- DistilBERT must contain `config.json` and `tokenizer/` files.
- SHA-256 weight checksums must match the recorded hashes.

### Environment Variable Overrides

| Variable | Default | Purpose |
|---|---|---|
| `TSD_REGISTRY` | `artifacts/active_model.json` | Path to active model registry file |
| `TSD_CHECKPOINT` | None | Explicit override path to a bundle directory; bypasses registry |
| `TSD_DEVICE` | `cpu` | Device for inference (`cpu`, `cuda`, `auto`) |
| `API_URL` | `http://localhost:8000` | Target URL used by Streamlit frontend |
| `HF_HUB_OFFLINE` | `1` in Docker | Prevents external HuggingFace network requests |

---

## 2. Launch Mode: Python CLI

The CLI provides quick inference without running any network servers:

```bash
# Single text prediction
uv run python -m app.prediction "The quarterly revenue increased by 14% year-over-year."

# Batch prediction with specified batch size
uv run python -m app.prediction --batch-size 4 "Hello!" "Please submit the invoice immediately."
```

Output includes predicted style and tone, calibrated confidence scores, token statistics, and any heuristics warnings.

---

## 3. Launch Mode: FastAPI Backend

Run the backend server locally using Uvicorn:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Accessing Backend Services

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
- **Model Metadata:** [http://127.0.0.1:8000/model/info](http://127.0.0.1:8000/model/info)
- **Readiness Probe:** [http://127.0.0.1:8000/health/ready](http://127.0.0.1:8000/health/ready)
- **Liveness Probe:** [http://127.0.0.1:8000/health/live](http://127.0.0.1:8000/health/live)

---

## 4. Launch Mode: Streamlit Frontend

In a separate terminal, launch the interactive web interface:

```bash
uv run streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Features:
- Single text analysis with interactive confidence gauges, score distribution charts, and warning alerts.
- Batch processing via text area input or UTF-8 CSV file upload (must have a `text` column).
- Export batch predictions as CSV or JSON.

---

## 5. Launch Mode: Docker Compose

Docker Compose coordinates three containers:
1. `backend`: FastAPI running on Python 3.13 slim with read-only `./artifacts` mount.
2. `frontend`: Streamlit UI talking to `http://backend:8000`.
3. `nginx`: Reverse proxy on host port 8000 routing UI and API requests.

### Starting Docker Compose

Make sure a model is selected on the host first, then run:

```bash
# Build and start services in background
docker compose up --build -d

# Check service status
docker compose ps

# View backend logs
docker compose logs -f backend
```

Access the entire application via Nginx at **http://localhost:8000**:
- Web UI: `http://localhost:8000/`
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health/ready`

### Switching Models in Docker

1. Run model selection on the host:
   ```bash
   uv run python -m scripts.select_model --checkpoint artifacts/experiments/bilstm_01
   ```
2. Restart the backend container:
   ```bash
   docker compose restart backend
   ```
