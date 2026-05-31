# Deployment Guide

## Docker Compose (Recommended)

The recommended way to deploy the application is using Docker Compose, which starts all three services (backend, frontend, Nginx) with a single command.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (≥ 20.10)
- [Docker Compose](https://docs.docker.com/compose/install/) (≥ 2.0)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/zave52/text-style-detection.git
    cd text-style-detection
    ```

2. **Build and start all services:**

    ```bash
    docker compose up --build
    ```

    !!! info "First run"
        The first build may take several minutes as Docker downloads base images and installs Python dependencies. The backend will also download the spaCy `en_core_web_sm` model on first startup.

3. **Access the application:**

    Open your browser and navigate to **http://localhost:8000**.

    | Path | Service | Description |
    |------|---------|-------------|
    | `/` | Streamlit | Interactive web UI for text classification |
    | `/predict` | FastAPI | REST API endpoint |
    | `/docs` | Swagger UI | Interactive API documentation |
    | `/openapi.json` | FastAPI | OpenAPI specification |

4. **Stop the application:**

    ```bash
    docker compose down
    ```

### Running in Background

To run the services in detached mode:

```bash
docker compose up --build -d
```

View logs:

```bash
docker compose logs -f
```

---

## Local Development Setup

For developing the notebooks and retraining models (the application itself should be run via Docker):

### Prerequisites

- Python ≥ 3.10
- pip

### Steps

1. **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate     # Linux/macOS
    # .venv\Scripts\activate      # Windows
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3. **(Optional) Rebuild the dataset from raw text files:**

    ```bash
    cd scripts
    python build_dataset.py
    cd ..
    ```

4. **(Optional) Retrain models:**

    ```bash
    jupyter notebook
    ```

    Execute the notebooks sequentially:

    1. `eda_01.ipynb` — Exploratory Data Analysis
    2. `modelling_02.ipynb` — TF-IDF + LinearSVC training
    3. `embedding_models_03.ipynb` — Embeddings + MultiOutput training

!!! warning "Application deployment"
    The FastAPI backend and Streamlit frontend are designed to run inside Docker containers. Use `docker compose up --build` to deploy the full application.

---

## Port Configuration

| Service | Internal Port | External Port | Configurable |
|---------|--------------|---------------|-------------|
| Backend (FastAPI) | 8000 | — (not exposed) | No |
| Frontend (Streamlit) | 8501 | — (not exposed) | No |
| Nginx | 80 | **8000** | Yes (in `docker-compose.yml`) |

To change the external port, modify the `ports` mapping in `docker-compose.yml`:

```yaml
nginx:
  ports:
    - "3000:80"  # Change 8000 to your desired port
```

---

## Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `API_URL` | Frontend | `http://localhost:8000` | Backend API base URL. Set to `http://backend:8000` in Docker Compose |

---

## Troubleshooting

### Backend fails to start

**Symptom:** Backend container exits immediately or logs show model loading errors.

**Possible causes:**

- Missing `saving/style_model.joblib` or `saving/tone_model.joblib` files
- Insufficient memory for loading models

**Solution:** Ensure model files exist in the `saving/` directory. If missing, retrain by running `modelling_02.ipynb`.

### Frontend can't connect to backend

**Symptom:** "Error connecting to the backend" message in the Streamlit UI.

**Possible causes:**

- Backend hasn't finished starting yet (model loading takes time)
- `API_URL` environment variable is incorrect

**Solution:** Wait for the backend to finish loading models (check logs with `docker compose logs backend`). Ensure `API_URL=http://backend:8000` is set in `docker-compose.yml`.

### Port already in use

**Symptom:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:** Change the external port in `docker-compose.yml` or stop the conflicting service:

```bash
# Find what's using port 8000
lsof -i :8000

# Or change the port in docker-compose.yml
```
