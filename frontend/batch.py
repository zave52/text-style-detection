"""CSV input and downloadable batch result helpers for the Streamlit frontend."""

import csv
import io
import json


def parse_text_csv(content: bytes) -> list[str]:
    """Read message text from a UTF-8 CSV with a required ``text`` column."""
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV files must use UTF-8 encoding.") from error

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if not reader.fieldnames or "text" not in reader.fieldnames:
        raise ValueError("CSV files must contain a column named 'text'.")

    return [(row.get("text") or "") for row in reader if row is not None]


def batch_results_csv(messages: list[str], results: list[dict]) -> str:
    """Serialize predictions with their source messages to a UTF-8 CSV string."""
    output = io.StringIO(newline="")
    fields = [
        "text",
        "model",
        "style",
        "style_confidence",
        "style_scores",
        "tone",
        "tone_confidence",
        "tone_scores",
        "character_count",
        "token_count",
        "used_token_count",
        "max_length",
        "truncated",
        "unknown_token_count",
        "unknown_ratio",
        "warnings",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for message, result in zip(messages, results, strict=True):
        writer.writerow(
            {
                "text": message,
                **{
                    field: json.dumps(result[field], ensure_ascii=False)
                    if field in {"style_scores", "tone_scores", "warnings"}
                    else result[field]
                    for field in fields
                    if field != "text"
                },
            }
        )
    return output.getvalue()
