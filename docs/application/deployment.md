# Deployment Guide

Deploy a **complete calibrated neural bundle** selected through the active-model registry. The application serves TextCNN, BiLSTM, or DistilBERT using saved preprocessing and calibration. The original joblib classifiers are historical baselines and are not required for serving.

## Prepare and select a model

Run commands from the repository root. Dependencies are locked in `uv.lock`; [uv](https://docs.astral.sh/uv/getting-started/installation/) installs the configured Python 3.13 environment and CPU PyTorch:

```bash
uv sync --frozen
```

If you already have a calibrated shared-pipeline experiment, select it:

```bash
uv run python -m scripts.select_model --checkpoint artifacts/experiments/distilbert_01
```

Selection verifies the bundle and records its relative path, architecture, and pinned checksums in `artifacts/active_model.json`. The selected directory must remain under `artifacts/` for the default Compose mount. Copy the entire directory when transferring a trained model; `best.pt` alone is insufficient.

A complete bundle contains `bundle.json`, `best.pt`, `calibration.json`, `labels.json`, `model_config.json`, `training_metadata.json`, `training_config.json`, `data_manifest.json`, `history.json`, and `summary.json`. Word models also include `vocabulary.json`. DistilBERT includes `config.json` and its `tokenizer/` files. Keep any additional files listed in `bundle.json` as well.

If no trained experiment exists, prepare the data and use the shared training command with a new directory:

```bash
uv run python -m scripts.build_dataset
uv run python -m scripts.train --model textcnn --run-dir artifacts/experiments/textcnn_01 --epochs 20
uv run python -m scripts.select_model --checkpoint artifacts/experiments/textcnn_01
```

Shared-pipeline training performs validation-only temperature calibration and seals the resulting bundle. For an older unsealed shared-pipeline run, use `scripts.calibrate` with its matching prepared dataset before selection. Do not overwrite or delete a valid sealed bundle to bypass integrity errors.

## Local development without Docker

Start the API:

```bash
uv run uvicorn app.main:app --reload
```

Start Streamlit in another terminal:

```bash
uv run streamlit run frontend/app.py
```

| Address | Purpose |
|---|---|
| `http://localhost:8501` | Streamlit UI |
| `http://localhost:8000/docs` | Interactive API documentation |
| `http://localhost:8000/health/ready` | Model readiness |
| `http://localhost:8000/model/info` | Loaded model information |

Streamlit uses `API_URL`, defaulting to `http://localhost:8000`. It only requires its own code, Streamlit, and requests; the API validates text length and batch size and returns `422` errors for the UI to display.

## Docker Compose

Install Docker with Compose v2. Prepare the selected model on the host first, then run:

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

Open **http://localhost:8000** for the UI and `/docs` for Swagger. Health, model information, and all prediction paths are also forwarded by Nginx. Only Nginx publishes a host port.

### Model mount and readiness

The backend uses this bind mount:

```yaml
volumes:
  - type: bind
    source: ./artifacts
    target: /app/artifacts
    read_only: true
    bind:
      create_host_path: false
```

Compose configures `TSD_REGISTRY=/app/artifacts/active_model.json`. The selected bundle and registry must be inside this mounted tree with their relative layout preserved. The host directory must already exist. Weights are not copied into the image and are excluded from the build context along with historical `saving/` artifacts.

The backend Dockerfile checks `http://127.0.0.1:8000/health/ready` using Python's standard library, so no curl installation is required. Checks run every 10 seconds, with a 6-second timeout, six retries, and a 60-second initialization grace period. The frontend and proxy depend on the backend being healthy.

If the registry, bundle, tokenizer, calibration, or checksums are invalid, the process logs the loading error and readiness returns `503`. Compose reports the backend as unhealthy and does not start its dependent services. Fix the files or model selection and restart; the process does not repeatedly reload a failed checkpoint on its own.

The backend sets `HF_HUB_OFFLINE=1`. DistilBERT serving uses only the saved encoder configuration, weights, and local tokenizer. Dependency installation during image build still requires access to the configured package sources.

### Switching a running model

Run selection on the host, where the registry is writable:

```bash
uv run python -m scripts.select_model --checkpoint artifacts/experiments/bilstm_01
docker compose restart backend
```

If initial startup stopped before creating the UI/proxy because no model was available, run `docker compose up -d` after fixing selection to start the remaining services. If `TSD_CHECKPOINT` is set, it takes precedence over the registry; unset it before switching through the registry.

Do not run calibration, training, or model selection inside the serving container. Its artifact mount is read-only, and the image contains the API package rather than training scripts or datasets.

### Logs and shutdown

```bash
docker compose logs -f backend
docker compose logs -f frontend nginx
docker compose down
```

`docker compose down` removes the containers and network. The host `artifacts/` directory remains available for the next deployment.

## Environment variables

| Variable | Local default | Compose behavior |
|---|---|---|
| `API_URL` | `http://localhost:8000` | Frontend uses `http://backend:8000` |
| `TSD_DEVICE` | `cpu` | Passed to the backend, default `cpu`; shipped PyTorch wheels are CPU builds |
| `TSD_REGISTRY` | `artifacts/active_model.json` under the repository | Set to `/app/artifacts/active_model.json` |
| `TSD_CHECKPOINT` | Unset; use the active registry | Optional explicit **container** bundle path, such as `/app/artifacts/experiments/distilbert_01` |
| `HF_HUB_OFFLINE` | Not required for saved local bundles | Set to `1` in the backend image |

When setting `TSD_CHECKPOINT` for Docker, do not pass a Windows host path; use the path under `/app/artifacts`. All consumers cache the model in their process, so restart the API after selection or configuration changes.

## Troubleshooting

| Symptom | Check and action |
|---|---|
| Bind mount source is missing | Prepare the host `artifacts/` directory and select a model before Compose startup |
| Backend unhealthy or dependency startup fails | Run `docker compose logs backend`; verify that `active_model.json` and the full selected bundle are mounted |
| Registry references a path outside the mount | Move the complete bundle under host `artifacts/experiments/` and select it again on the host |
| Checksum or tokenizer error | Restore the complete original bundle; avoid editing sealed files or copying only weights |
| Newly selected model is not used | Unset an explicit `TSD_CHECKPOINT` override and restart the API |
| UI reports connection failure | Confirm the backend is healthy and `API_URL` points to the API, not the UI |
| UI displays a validation error | Follow the API's `422` details; messages must be non-whitespace strings of 1–2000 characters and batches must contain 1–64 messages |
| Port 8000 is occupied | Change Nginx's host port mapping, for example `3000:80`; internal service addresses remain unchanged |

See the [API reference](api.md) for response schemas and the [architecture guide](architecture.md) for loading and request flow.
