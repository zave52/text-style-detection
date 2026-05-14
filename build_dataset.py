import os
import pandas as pd

base_path = "dataset"

data = []

for style in os.listdir(base_path):

    style_path = os.path.join(base_path, style)

    if os.path.isdir(style_path):

        for tone in os.listdir(style_path):

            tone_path = os.path.join(style_path, tone)

            if os.path.isdir(tone_path):

                for file in os.listdir(tone_path):

                    if file.endswith(".txt"):

                        file_path = os.path.join(
                            tone_path,
                            file
                        )

                        try:
                            with open(
                                file_path,
                                "r",
                                encoding="utf-8"
                            ) as f:

                                text = f.read().strip()

                            data.append({
                                "text": text,
                                "style": style,
                                "tone": tone
                            })

                        except Exception as e:
                            print(
                                f"Помилка {file}: {e}"
                            )

df = pd.DataFrame(data)

df = df.sample(
    frac=1,
    random_state=42
)

df.to_csv(
    "dataset.csv",
    index=False,
    encoding="utf-8"
)

print(
    f"Готово! Записано {len(df)} рядків"
)