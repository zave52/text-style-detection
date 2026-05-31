import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Style & Tone Detector", page_icon="🎭")

st.title("🎭 Text Style & Tone Detection")
st.markdown("Enter your text below to predict its style and tone.")

user_text = st.text_area(
    "Your Text:",
    height=150,
    placeholder="Type something here..."
)

col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    btn_all = st.button("Predict Both (Style & Tone)", use_container_width=True)

with col2:
    btn_style = st.button("Style Only", use_container_width=True)

with col3:
    btn_tone = st.button("Tone Only", use_container_width=True)


def handle_prediction(
    endpoint: str,
    text: str,
    spinner_msg: str,
    success_msg: str,
    keys_to_show: list
):
    if not text.strip():
        st.warning("Please enter some text first.")
        return

    with st.spinner(spinner_msg):
        try:
            response = requests.post(
                f"{API_URL}{endpoint}",
                json={"text": text}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(success_msg)
                for key in keys_to_show:
                    st.write(f"**{key.capitalize()}:** {result[key]}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception:
            st.error("Error connecting to the backend")


if btn_all:
    handle_prediction(
        "/predict",
        user_text,
        "Analyzing...",
        "Style and Tone Predicted",
        ["style", "tone"]
    )

if btn_style:
    handle_prediction(
        "/predict/style",
        user_text,
        "Analyzing style...",
        "Style Predicted",
        ["style"]
    )

if btn_tone:
    handle_prediction(
        "/predict/tone",
        user_text,
        "Analyzing tone...",
        "Tone Predicted",
        ["tone"]
    )
