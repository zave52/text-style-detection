from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


df = pd.read_csv("../data/dataset.csv")

print(df.head())

X = df["clean_text"]

y_style = df["style"]
y_tone = df["tone"]


X_train_style, X_test_style, y_train_style, y_test_style = train_test_split(
    X,
    y_style,
    test_size=0.2,
    random_state=42,
    stratify=y_style
)

X_train_tone, X_test_tone, y_train_tone, y_test_tone = train_test_split(
    X,
    y_tone,
    test_size=0.2,
    random_state=42,
    stratify=y_tone
)

style_model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)

style_model.fit(X_train_style, y_train_style)

style_predictions = style_model.predict(X_test_style)

style_f1 = f1_score(
    y_test_style,
    style_predictions,
    average="weighted"
)

print("\nSTYLE CLASSIFICATION")
print(f"F1-score: {style_f1:.4f}")

print(
    classification_report(
        y_test_style,
        style_predictions
    )
)

tone_model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)

tone_model.fit(X_train_tone, y_train_tone)

tone_predictions = tone_model.predict(X_test_tone)

tone_f1 = f1_score(
    y_test_tone,
    tone_predictions,
    average="weighted"
)

print("\nTONE CLASSIFICATION")
print(f"F1-score: {tone_f1:.4f}")

print(
    classification_report(
        y_test_tone,
        tone_predictions
    )
)