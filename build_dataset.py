from pathlib import Path
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

base_path = Path("dataset")

data = []

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words)


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
