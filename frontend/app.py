"""Streamlit interface consuming the shared prediction response contract."""

import json
import os

import requests
import streamlit as st
from batch import batch_results_csv, parse_text_csv

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Style & Tone Detector", page_icon="🎭")
st.title("🎭 Text Style & Tone Detection")
st.markdown("Analyze English messages with the selected model.")


def request_prediction(endpoint: str, payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=(5, 120))
        if not response.ok:
            detail = response.json().get("detail", "Prediction request failed.")
            if isinstance(detail, list):
                detail = "; ".join(item.get("msg", "Invalid input") for item in detail)
            st.error(str(detail))
            return None
        return response.json()
    except requests.RequestException:
        st.error("Cannot reach the prediction API. Check that the backend is running.")
    except ValueError:
        st.error("The backend returned an invalid response.")
    return None


def display_prediction(result: dict, tasks: tuple[str, ...]) -> None:
    for column, task in zip(st.columns(len(tasks)), tasks, strict=True):
        with column:
            st.metric(task.capitalize(), result[task])
            st.caption(f"Confidence: {result[f'{task}_confidence']:.1%}")
            st.dataframe(
                [
                    {"class": label, "score": score}
                    for label, score in result[f"{task}_scores"].items()
                ],
                hide_index=True,
            )
    st.caption(
        f"Model: {result['model']} · Tokens: {result['used_token_count']}/{result['token_count']} "
        f"· Limit: {result['max_length']} · Unknown content tokens: {result['unknown_ratio']:.1%}"
    )
    for warning in result["warnings"]:
        if warning["task"] is None or warning["task"] in tasks:
            st.warning(warning["message"])
    with st.expander("Full prediction details"):
        st.json(result)
        st.download_button(
            "Download prediction JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="text-style-prediction.json",
            mime="application/json",
            key="single_json_download",
        )


single_tab, batch_tab = st.tabs(["Single message", "Batch messages"])
with single_tab:
    text = st.text_area("Your text", height=150, key="single_text")
    selected = None
    for column, task, title in zip(
        st.columns(3),
        ("both", "style", "tone"),
        ("Predict both", "Style only", "Tone only"),
        strict=True,
    ):
        if column.button(title, key=task):
            selected = task
    if selected:
        endpoint = "/predict" if selected == "both" else f"/predict/{selected}"
        with st.spinner("Analyzing..."):
            prediction = request_prediction(endpoint, {"text": text})
        if prediction is not None:
            display_prediction(prediction, ("style", "tone") if selected == "both" else (selected,))

with batch_tab:
    st.caption(
        "Paste one message per line or upload a UTF-8 CSV with a 'text' column. "
        "Blank lines in pasted input are ignored; the API validates batch limits. "
        "An uploaded CSV takes precedence over pasted messages."
    )
    batch_text = st.text_area("Messages", height=200, key="batch_text")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if st.button("Predict batch", key="batch"):
        try:
            messages = (
                parse_text_csv(uploaded_csv.getvalue())
                if uploaded_csv is not None
                else [line for line in batch_text.splitlines() if line.strip()]
            )
        except ValueError as error:
            st.error(str(error))
        else:
            with st.spinner("Analyzing messages..."):
                response = request_prediction("/predict/batch", {"texts": messages})
            if response is not None:
                st.dataframe(
                    [
                        {
                            "text": text,
                            "style": row["style"],
                            "tone": row["tone"],
                            "style_confidence": row["style_confidence"],
                            "tone_confidence": row["tone_confidence"],
                            "warnings": "; ".join(
                                warning["message"] for warning in row["warnings"]
                            ),
                        }
                        for text, row in zip(messages, response["results"], strict=True)
                    ],
                    hide_index=True,
                )
                st.download_button(
                    "Download batch CSV",
                    data=batch_results_csv(messages, response["results"]),
                    file_name="text-style-predictions.csv",
                    mime="text/csv",
                    key="batch_csv_download",
                )
                st.download_button(
                    "Download batch JSON",
                    data=json.dumps(response, ensure_ascii=False, indent=2),
                    file_name="text-style-predictions.json",
                    mime="application/json",
                    key="batch_json_download",
                )
                with st.expander("Full batch prediction details"):
                    st.json(response)
