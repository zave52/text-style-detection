import re
from pathlib import Path

import spacy
import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

base_path = Path("dataset")

data = []

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)

    text = text.lower()

    doc = nlp(text)

    clean_tokens = []

    for token in doc:
        if token.is_space:
            continue

        if token.is_punct:
            clean_tokens.append(token.text)
        else:
            lemma = token.lemma_.strip()

            if lemma and lemma not in STOP_WORDS:
                clean_tokens.append(lemma)

    return " ".join(clean_tokens)


for style_path in base_path.iterdir():

    if style_path.is_dir():
        style = style_path.name

        for tone_path in style_path.iterdir():

            if tone_path.is_dir():
                tone = tone_path.name

                for file_path in tone_path.glob("*.txt"):
                    try:
                        raw_text = file_path.read_text(encoding="utf-8").strip()

                        clean_text = preprocess_text(raw_text)
                        
                        if not clean_text.strip():
                            continue

                        data.append(
                            {
                                "text": raw_text,
                                "clean_text": clean_text,
                                "style": style,
                                "tone": tone
                            }
                        )

                    except Exception as e:
                        print(f"Error {file_path.name}: {e}")

df = pd.DataFrame(data)

df = df.sample(frac=1,random_state=42).reset_index(drop=True)

output_file = Path("dataset.csv")

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print(f"Done! Wrote {len(df)} rows to {output_file}")
