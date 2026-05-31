---
hide:
  - navigation
---

# Text Style & Tone Detection

A machine learning system for classifying the **writing style** and **emotional tone** of English text.

The project includes an end-to-end ML pipeline (EDA → modeling → evaluation), a FastAPI inference backend, and a Streamlit web interface — all containerized with Docker Compose.

---

## Problem Statement

Given an arbitrary English text, the system predicts two independent properties:

| Property | Classes | Description |
|----------|---------|-------------|
| **Style** | `academic`, `business`, `formal`, `informal`, `literary` | The structural and lexical register of the text |
| **Tone** | `aggressive`, `friendly`, `neutral`, `sarcastic`, `urgent` | The emotional coloring conveyed by the author |

These two dimensions are orthogonal — any style can appear with any tone (e.g., an `academic` text can be `sarcastic`, or a `business` text can be `aggressive`).

---

## Key Features

<div class="grid cards" markdown>

-   **Exploratory Data Analysis**

    ---

    Label distributions, text length statistics, cross-tabulations, and visualizations across all 25 style × tone combinations.

-   **Two Modeling Approaches**

    ---

    TF-IDF + LinearSVC for sparse features, and spaCy embeddings + SVM/Random Forest for dense representations.

-   **Hyperparameter Tuning**

    ---

    `GridSearchCV` with 5-fold stratified cross-validation and F1-weighted scoring for rigorous model selection.

-   **REST API + Web UI**

    ---

    FastAPI backend with Swagger docs and a Streamlit interactive frontend, containerized with Docker Compose.

</div>

---

## Project Structure

```
text-style-detection/
├── app/                          # FastAPI backend
│   ├── main.py                   #   Entrypoint & route definitions
│   ├── ml.py                     #   Model loading, preprocessing & inference
│   └── schemas.py                #   Pydantic request/response schemas
├── frontend/
│   └── app.py                    # Streamlit web interface
├── scripts/
│   └── build_dataset.py          # Build dataset.csv from raw text files
├── dataset/                      # Raw text corpus (style/tone/sample.txt)
│   ├── academic/
│   ├── business/
│   ├── formal/
│   ├── informal/
│   └── literary/
├── data/                         # Processed data & saved pipelines
│   └── dataset.csv               #   Compiled dataset (1 000 samples)
├── saving/                       # Trained TF-IDF + LinearSVC models
│   ├── style_model.joblib        #   (not committed to git)
│   └── tone_model.joblib         #   (not committed to git)
├── nginx/
│   └── nginx.conf                # Nginx reverse proxy configuration
├── eda_01.ipynb                  # Notebook 01 — EDA
├── modelling_02.ipynb            # Notebook 02 — TF-IDF + LinearSVC
├── embedding_models_03.ipynb     # Notebook 03 — Embeddings + MultiOutput
├── requirements.txt
├── docker-compose.yml
├── backend.Dockerfile
└── frontend.Dockerfile
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)

### Deploy with Docker

```bash
git clone https://github.com/zave52/text-style-detection.git
cd text-style-detection
docker compose up --build -d
```

The application will be available at **http://localhost:8000**:

| Path | Service |
|------|---------|
| `/` | Streamlit frontend |
| `/predict` | FastAPI prediction endpoint |
| `/docs` | Swagger UI (interactive API docs) |

For more details, see the [Deployment Guide](application/deployment.md).

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML / NLP | scikit-learn, spaCy, pandas, seaborn, matplotlib |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose, Nginx |
| Serialization | joblib |
