import requests
import streamlit as st
import uuid

# Create a stable session id once per browser session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())


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
        endpoint = "/evaluate" if "Evaluate" in mode else "/fix" if "Grammar" in mode else "/score"
        headers = {"X-Session-Id": st.session_state["session_id"]}

        try:
            resp = requests.post(
                f"{API}{endpoint}",
                json={"text": text},
                headers=headers,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            if endpoint == "/score":
                st.subheader("CEFR Result")
                st.write(f"**Level:** {data.get('level')}")
                st.write(f"**Score:** {data.get('score')}/100")
                st.write("**Rationale:**")
                st.code(data.get("rationale", ""), language="markdown")
            else:
                st.subheader("Response")
                st.code(data.get("result", ""), language="markdown")

        except Exception as e:
            st.error(f"Request failed: {e}")
            st.info("Make sure the FastAPI server is running:\n  uvicorn src.api:app --reload")
