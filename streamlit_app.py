import requests
import streamlit as st

st.set_page_config(page_title="NorskAgent", page_icon="🇳🇴", layout="centered")

API = "http://127.0.0.1:8000"  # FastAPI base URL

st.title("NorskAgent 🇳🇴")
mode = st.radio("Choose mode:", ["Evaluate (exam-style)", "Grammar Fix", "CEFR Score"], horizontal=False)

text = st.text_area("Write a Norwegian sentence:", height=120, placeholder="Jer er trott")

col1, col2 = st.columns([1,1])
with col1:
    run = st.button("Run")
with col2:
    clear = st.button("Clear")

if clear:
    st.experimental_rerun()

if run:
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        try:
            endpoint = "/evaluate" if "Evaluate" in mode else "/fix" if "Grammar" in mode else "/score"
            resp = requests.post(f"{API}{endpoint}", json={"text": text}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.subheader("Response")
            st.code(data.get("result", ""), language="markdown")
        except Exception as e:
            st.error(f"Request failed: {e}\nMake sure the FastAPI server is running.")
            st.info("Start it with:  uvicorn src.api:app --reload")
