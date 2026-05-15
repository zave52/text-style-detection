from pathlib import Path
import pandas as pd

base_path = Path("dataset")

data = []

for style_path in base_path.iterdir():

    if style_path.is_dir():
        style = style_path.name

        for tone_path in style_path.iterdir():

            if tone_path.is_dir():
                tone = tone_path.name

                for file_path in tone_path.glob("*.txt"):
                    try:
                        text = file_path.read_text(encoding="utf-8").strip()

                        data.append(
                            {
                                "text": text,
                                "style": style,
                                "tone": tone
                            }
                        )

                    except Exception as e:
                        print(f"Error {file_path.name}: {e}")

df = pd.DataFrame(data)

df = df.sample(
    frac=1,
    random_state=42
)

output_file = Path("dataset.csv")

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print(f"Done! Wrote {len(df)} rows to {output_file}")
