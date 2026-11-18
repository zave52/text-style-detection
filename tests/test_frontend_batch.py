import csv
import io

import pytest

from frontend.batch import batch_results_csv, parse_text_csv


def test_parse_text_csv_reads_utf8_bom_and_quoted_newlines():
    content = '\ufefftext,source\n"Hello, world!",one\n"Second\nmessage",two\n'.encode("utf-8")

    assert parse_text_csv(content) == ["Hello, world!", "Second\nmessage"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"message\nhello\n", "column named 'text'"),
        (b"text\n\xff\n", "UTF-8 encoding"),
    ],
)
def test_parse_text_csv_rejects_invalid_uploads(content, message):
    with pytest.raises(ValueError, match=message):
        parse_text_csv(content)


def test_batch_results_csv_keeps_source_text_and_structured_fields():
    results = [
        {
            "model": "textcnn",
            "style": "formal",
            "style_confidence": 0.8,
            "style_scores": {"formal": 0.8},
            "tone": "neutral",
            "tone_confidence": 0.7,
            "tone_scores": {"neutral": 0.7},
            "character_count": 13,
            "token_count": 3,
            "used_token_count": 3,
            "max_length": 128,
            "truncated": False,
            "unknown_token_count": 0,
            "unknown_ratio": 0.0,
            "warnings": [{"code": "uncertain_prediction", "message": "Check scores."}],
        }
    ]

    rows = list(csv.DictReader(io.StringIO(batch_results_csv(["Hello, world!"], results))))

    assert rows[0]["text"] == "Hello, world!"
    assert rows[0]["style"] == "formal"
    assert rows[0]["style_scores"] == '{"formal": 0.8}'
    assert rows[0]["warnings"] == '[{"code": "uncertain_prediction", "message": "Check scores."}]'
