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

if btn_all:
    if user_text.strip():
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Style and Tone Predicted")
                    st.write(f"**Style:** {result['style']}")
                    st.write(f"**Tone:** {result['tone']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error("Error connecting to the backend. Is FastAPI running?")
    else:
        st.warning("Please enter some text first.")

if btn_style:
    if user_text.strip():
        with st.spinner("Analyzing style..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict/style",
                    json={"text": user_text}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Style Predicted")
                    st.write(f"**Style:** {result['style']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error("Error connecting to the backend.")
    else:
        st.warning("Please enter some text first.")

if btn_tone:
    if user_text.strip():
        with st.spinner("Analyzing tone..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict/tone",
                    json={"text": user_text}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Tone Predicted")
                    st.write(f"**Tone:** {result['tone']}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error("Error connecting to the backend.")
    else:
        st.warning("Please enter some text first.")
