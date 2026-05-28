# Text Style Detection

A machine learning project for detecting and classifying text styles using NLP techniques. The pipeline covers exploratory data analysis, feature engineering with spaCy, and supervised classification with scikit-learn.

---

## Project Structure

```
text-style-detection/
├── app/                        # Backend application
│   ├── __pycache__/
│   ├── main.py                 # Entry point (FastAPI / Flask app)
│   ├── ml.py                   # ML inference logic
│   └── schemas.py              # Request/response schemas
│
├── data/
│   └── dataset.csv             # Raw dataset
│
├── dataset/                    # Labelled text samples by style and tone
│   ├── academic/
│   │   ├── aggressive/
│   │   ├── friendly/
│   │   ├── neutral/
│   │   ├── sarcastic/
│   │   └── urgent/
│   ├── business/
│   │   ├── aggressive/
│   │   ├── friendly/
│   │   ├── neutral/
│   │   ├── sarcastic/
│   │   └── urgent/
│   ├── formal/
│   │   ├── aggressive/
│   │   ├── friendly/
│   │   ├── neutral/
│   │   ├── sarcastic/
│   │   └── urgent/
│   ├── informal/
│   │   ├── aggressive/
│   │   ├── friendly/
│   │   ├── neutral/
│   │   ├── sarcastic/
│   │   └── urgent/
│   └── literaly/
│       ├── aggressive/
│       ├── friendly/
│       ├── neutral/
│       ├── sarcastic/
│       └── urgent/
│
├── frontend/
│   └── app.py                  # Streamlit (or similar) frontend
├── nginx/
│   └── nginx.conf              # Reverse proxy config
├── saving/                     # Persisted trained models
│
├── scripts/
│   └── build_dataset.py        # Dataset construction script
├── venv/                       # Virtual environment (not committed)
├── .dockerignore
├── .gitignore
├── backend.Dockerfile
├── docker-compose.yml
├── eda_01.ipynb                # Exploratory Data Analysis
├── embedding_models_03.ipynb   # Embedding model experiments
├── frontend.Dockerfile
├── modelling_02.ipynb          # Model training and evaluation
├── README.md
└── requirements.txt
```

---

## Dataset
The dataset/ directory contains labelled text samples organised by style (e.g. academic, business, formal, informal, literaly) and tone (e.g. aggressive, friendly, neutral, sarcastic, urgent).
Each tone folder contains 40 text files with lengths ranging from 20 to 800 characters, increasing in steps of ~20 characters between adjacent files. This ensures the models are trained on a diverse range of text lengths and are not biased towards a particular input size.

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/zave52/text-style-detection.git
cd text-style-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv

source venv/bin/activate

venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the required spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Running the Application
To start the backend application execute:

```bash
uvicorn app.main:app --reload
```

In a separate terminal, start the frontend by executing:

```bash
streamlit run frontend/app.py
```

The application will open automatically in your browser.

---

## Dependencies

| Package | Version |
|---|---|
| pandas | 3.0.3 |
| spacy | 3.8.13 |
| seaborn | 0.13.2 |
| scikit-learn | ≥ 1.5.0 |

---

## Notebooks

### `eda_01.ipynb` — Exploratory Data Analysis
Investigates the dataset structure, class distribution, text length statistics, and linguistic features. Includes visualizations built with seaborn to understand the characteristics of each text style and tone.

### `modelling_02.ipynb` — Modelling
Builds and evaluates classification models for text style and tone detection. Uses spaCy for text preprocessing and feature extraction, and scikit-learn for training and evaluating classifiers. Saves the final models as `style_model.pkl` and `tone_model.pkl`.

### `embedding_models_03.ipynb` — Embedding Models
Experiments with text embedding approaches to improve feature representations for style and tone classification.

---

## Methodology

1. **Data loading & inspection** — load raw text samples with their style labels.
2. **EDA** — analyse class balance, token/sentence distributions, and linguistic patterns.
3. **Feature engineering** — extract NLP features using spaCy (POS tags, lemmas, named entities, syntactic structures).
4. **Modelling** — train and cross-validate scikit-learn classifiers; compare performance with accuracy, precision, recall, and F1-score.
5. **Evaluation** — interpret results with confusion matrices and feature importance analysis.

---

## Tech Stack

- **spaCy** — linguistic feature extraction and text preprocessing
- **scikit-learn** — machine learning pipeline, vectorisation, and classification
- **pandas** — data manipulation and analysis
- **seaborn** — statistical data visualisation

---

## Results

# Model Comparison (MultiOutputClassifier wrapping SVM (RBF))

============================================================
SVM — style (['academic' 'business' 'formal' 'informal' 'literaly'])
============================================================
              precision    recall  f1-score   support

    academic       0.95      0.88      0.91        40
    business       0.78      0.80      0.79        40
      formal       0.82      0.82      0.82        40
    informal       0.83      0.95      0.88        40
    literaly       1.00      0.90      0.95        40

    accuracy                           0.87       200
   macro avg       0.88      0.87      0.87       200
weighted avg       0.88      0.87      0.87       200

============================================================
SVM — tone (['aggressive' 'friendly' 'neutral' 'sarcastic' 'urgent'])
============================================================
              precision    recall  f1-score   support

  aggressive       0.83      0.94      0.88        36
    friendly       0.93      0.91      0.92        44
     neutral       0.85      0.82      0.84        40
   sarcastic       0.90      0.83      0.86        42
      urgent       0.95      0.95      0.95        38
...
    accuracy                           0.89       200
   macro avg       0.89      0.89      0.89       200
weighted avg       0.89      0.89      0.89       200

# MultiOutputClassifier wrapping Random Forest

============================================================
RF — style (['academic' 'business' 'formal' 'informal' 'literaly'])
============================================================
              precision    recall  f1-score   support

    academic       0.88      0.93      0.90        40
    business       0.78      0.80      0.79        40
      formal       0.85      0.72      0.78        40
    informal       0.84      0.90      0.87        40
    literaly       0.93      0.93      0.93        40

    accuracy                           0.85       200
   macro avg       0.86      0.86      0.85       200
weighted avg       0.86      0.85      0.85       200

============================================================
RF — tone (['aggressive' 'friendly' 'neutral' 'sarcastic' 'urgent'])
============================================================
              precision    recall  f1-score   support

  aggressive       0.85      0.92      0.88        36
    friendly       0.93      0.89      0.91        44
     neutral       0.82      0.82      0.82        40
   sarcastic       0.86      0.88      0.87        42
      urgent       0.94      0.89      0.92        38
...
    accuracy                           0.88       200
   macro avg       0.88      0.88      0.88       200
weighted avg       0.88      0.88      0.88       200

## Summary Table

Model Target  Accuracy  CV F1-macro  CV std
  SVM  style     0.870       0.8534  0.0215
  SVM   tone     0.890       0.8721  0.0156
   RF  style     0.855       0.8365  0.0187
   RF   tone     0.880       0.8461  0.0131

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an [issue](https://github.com/zave52/text-style-detection/issues) or submit a pull request.

---

## Developers
The ML was develop by Pavlo Molytovnyk and Zahar Savchyn

--Pavlo Molytovnyk - https://github.com/PavloMolytovnyk
--Zahar Savchyn - https://github.com/zave52

## License

This project is open source. Please check the repository for licence details.