# 🎭 Text Style & Tone Detection

A machine learning system for classifying the **writing style** and **emotional tone** of English text. The project includes an end-to-end ML pipeline (EDA → modeling → evaluation), a FastAPI inference backend, and a Streamlit web interface — all containerized with Docker Compose.

> **Live Demo:** [ec2-16-171-112-170.eu-north-1.compute.amazonaws.com](http://ec2-16-171-112-170.eu-north-1.compute.amazonaws.com)
>
> *Note: The demo is hosted on an AWS EC2 instance and may be unavailable at the time of viewing.*

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
  - [Approach 1 — TF-IDF + LinearSVC](#approach-1--tf-idf--linearsvc)
  - [Approach 2 — spaCy Embeddings + MultiOutputClassifier](#approach-2--spacy-embeddings--multioutputclassifier)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [License](#license)

## Overview

The goal of this project is to automatically detect two independent properties of a given text:

| Property | Classes |
|----------|---------|
| **Style** | `academic`, `business`, `formal`, `informal`, `literary` |
| **Tone** | `aggressive`, `friendly`, `neutral`, `sarcastic`, `urgent` |

Two modeling approaches are explored and compared across the analysis notebooks.

## Key Features

- **Exploratory Data Analysis** — label distributions, text length statistics, and cross-tabulations
- **Two modeling approaches** — TF-IDF + LinearSVC and spaCy embeddings + SVM/Random Forest
- **Hyperparameter tuning** — `GridSearchCV` with 5-fold stratified cross-validation
- **REST API** — FastAPI backend with `/predict`, `/predict/style`, and `/predict/tone` endpoints
- **Web UI** — Streamlit-based interactive frontend
- **Docker Compose** — one-command deployment with Nginx reverse proxy

## Project Structure

```
text-style-detection/
├── app/                          # FastAPI backend
│   ├── main.py                   #   Application entrypoint & route definitions
│   ├── ml.py                     #   Model loading, preprocessing & inference
│   └── schemas.py                #   Pydantic request/response schemas
├── frontend/
│   └── app.py                    # Streamlit web interface
├── scripts/
│   └── build_dataset.py          # Script to build dataset.csv from raw text files
├── dataset/                      # Raw text corpus (style/tone/sample.txt)
│   ├── academic/
│   │   ├── aggressive/
│   │   ├── friendly/
│   │   ├── neutral/
│   │   ├── sarcastic/
│   │   └── urgent/
│   ├── business/
│   ├── formal/
│   ├── informal/
│   └── literary/
├── data/                         # Processed data & saved pipelines
│   └── dataset.csv               #   Compiled dataset (1 000 samples)
├── saving/                       # Trained TF-IDF + LinearSVC models
│   ├── style_model.joblib        # (not commited to git)
│   └── tone_model.joblib         # (not commited to git)
├── nginx/
│   └── nginx.conf                # Nginx reverse proxy configuration
├── eda_01.ipynb                  # Notebook 01 — Exploratory Data Analysis
├── modelling_02.ipynb            # Notebook 02 — TF-IDF + LinearSVC modeling
├── embedding_models_03.ipynb     # Notebook 03 — spaCy embeddings + MultiOutput
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker Compose orchestration
├── backend.Dockerfile            # Backend container image
└── frontend.Dockerfile           # Frontend container image
```

## Dataset

The dataset contains **1 000 English text samples**, balanced across all classes:

- **5 styles × 5 tones = 25 class combinations**, each with 40 samples
- **200 samples per style**, **200 samples per tone**
- Raw texts are stored in `dataset/{style}/{tone}/*.txt`
- The compiled CSV (`data/dataset.csv`) includes both the original `text` and the preprocessed `clean_text`

### Text Preprocessing

The preprocessing pipeline (shared between `scripts/build_dataset.py` and `app/ml.py`):

1. Remove URLs (`http://…`, `www.…`)
2. Tokenize and lemmatize using **spaCy** (`en_core_web_sm`)
3. Preserve punctuation tokens; lowercase lemmatized words
4. Collapse multiple spaces

## Models

### Approach 1 — TF-IDF + LinearSVC

> 📓 Notebook: [`modelling_02.ipynb`](modelling_02.ipynb)

Two independent classifiers (one for style, one for tone) trained on `clean_text`:

- **Feature extraction:** `TfidfVectorizer`
- **Classifier:** `LinearSVC`
- **Tuning:** `GridSearchCV` (5-fold CV, `f1_weighted` scoring)
- **Search space:**
  - `max_features`: {3 000, 5 000}
  - `ngram_range`: {(1,1), (1,2), (1,3)}
  - `C`: {0.1, 1, 10}

### Approach 2 — spaCy Embeddings + MultiOutputClassifier

> 📓 Notebook: [`embedding_models_03.ipynb`](embedding_models_03.ipynb)

A single `MultiOutputClassifier` predicting both targets simultaneously:

- **Feature extraction:** 300-dimensional word vectors via **spaCy** (`en_core_web_md`), averaged per document
- **Classifiers compared:** `SVC` (RBF kernel, C=10) and `RandomForestClassifier` (300 estimators)
- **Evaluation:** 5-fold stratified CV with F1-macro scoring

## Results

### Approach 1 — TF-IDF + LinearSVC (Test Set)

| Target | Accuracy | F1 (weighted) | CV F1 (weighted) | Best Params |
|--------|----------|---------------|-------------------|-------------|
| **Style** | 0.89 | 0.886 | 0.918 | `C=1`, `max_features=5000`, `ngram_range=(1,3)` |
| **Tone** | 0.94 | 0.935 | 0.950 | `C=1`, `max_features=5000`, `ngram_range=(1,2)` |

<details>
<summary>Per-class metrics — Style (TF-IDF + LinearSVC)</summary>

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| academic | 0.97 | 0.93 | 0.95 | 40 |
| business | 0.71 | 0.85 | 0.77 | 40 |
| formal | 0.88 | 0.72 | 0.79 | 40 |
| informal | 0.90 | 0.95 | 0.93 | 40 |
| literary | 1.00 | 0.97 | 0.99 | 40 |

</details>

<details>
<summary>Per-class metrics — Tone (TF-IDF + LinearSVC)</summary>

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| aggressive | 1.00 | 0.90 | 0.95 | 40 |
| friendly | 0.93 | 0.93 | 0.93 | 40 |
| neutral | 0.91 | 0.97 | 0.94 | 40 |
| sarcastic | 0.88 | 0.93 | 0.90 | 40 |
| urgent | 0.97 | 0.95 | 0.96 | 40 |

</details>

### Approach 2 — spaCy Embeddings (5-Fold CV)

| Model | Target | Test Accuracy | CV F1-macro |
|-------|--------|---------------|-------------|
| **SVM** | Style | 0.870 | 0.853 ± 0.022 |
| **SVM** | Tone | 0.890 | 0.872 ± 0.016 |
| RF | Style | 0.855 | 0.837 ± 0.019 |
| RF | Tone | 0.880 | 0.846 ± 0.013 |

> **Conclusion:** TF-IDF + LinearSVC outperforms the embedding-based approach on this dataset. The TF-IDF models are used for the deployed inference API.

## Getting Started

### Prerequisites

- Python ≥ 3.10
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) (for containerized deployment)

### Local Development

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zave52/text-style-detection.git
   cd text-style-detection
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **(Optional) Rebuild the dataset from raw text files:**

   ```bash
   cd scripts
   python build_dataset.py
   cd ..
   ```

4. **(Optional) Retrain models — run the Jupyter notebooks in order:**

   ```bash
   jupyter notebook
   ```

   Execute the notebooks sequentially: `eda_01.ipynb` → `modelling_02.ipynb` → `embedding_models_03.ipynb`

### Docker Deployment

Launch the entire stack (backend + frontend + Nginx) with a single command:

```bash
docker compose up --build
```

The application will be available at **http://localhost:8000**:

| Path | Service |
|------|---------|
| `/` | Streamlit frontend |
| `/predict` | FastAPI prediction endpoint |
| `/docs` | Interactive API documentation (Swagger UI) |

## API Reference

### `POST /predict`

Predict both style and tone.

**Request:**

```json
{
  "text": "Your text here"
}
```

**Response:**

```json
{
  "style": "academic",
  "tone": "neutral"
}
```

### `POST /predict/style`

Predict style only.

**Response:**

```json
{
  "style": "formal"
}
```

### `POST /predict/tone`

Predict tone only.

**Response:**

```json
{
  "tone": "sarcastic"
}
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML / NLP | scikit-learn, spaCy, pandas, seaborn, matplotlib |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose, Nginx |
| Serialization | joblib |
