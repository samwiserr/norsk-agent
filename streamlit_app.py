import os
import json
import requests
import streamlit as st

# Try to use a hosted API if provided; otherwise run agents locally in-process
API_BASE = os.getenv("API_BASE")  # e.g. "https://your-api.fly.dev"
CLOUD_MODE = os.getenv("CLOUD_MODE", "0") == "1"

st.set_page_config(page_title="Norsk Agent", page_icon="🇳🇴", layout="centered")
st.title("🇳🇴 Norsk Agent")

st.caption(
    "Grammar corrections, exam-style feedback and CEFR scoring for Norwegian (A1–B1). "
    "Tip: Use proper diacritics (æ, ø, å)."
)

# ---------- Utilities ----------
def call_api(endpoint: str, payload: dict):
    """Call a hosted API if API_BASE is set."""
    url = f"{API_BASE}{endpoint}"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def safe_error(e: Exception) -> str:
    msg = str(e)
    if "127.0.0.1" in msg or "localhost" in msg:
        return (
            "Unable to reach a local API at 127.0.0.1. "
            "On Streamlit Cloud you should either set API_BASE to a public API URL, "
            "or let the app call OpenAI directly (CLOUD_MODE=1)."
        )
    return msg

# ---------- UI ----------
text = st.text_area("Skriv en setning på norsk:", height=120, placeholder="F.eks. Jer er trott")
mode = st.selectbox("Velg modus", ["fix", "evaluate", "score"], index=0)
go = st.button("Kjør")

# ---------- Execution paths ----------
if go and text.strip():
    try:
        if API_BASE:
            # Remote API path (only if you host FastAPI somewhere)
            if mode == "fix":
                data = call_api("/fix", {"text": text})
                st.code(data.get("result", ""), language="markdown")
            elif mode == "evaluate":
                data = call_api("/evaluate", {"text": text})
                st.code(data.get("result", ""), language="markdown")
            else:  # score
                data = call_api("/score", {"text": text})
                st.json(data)
        else:
            # In-process path (recommended on Streamlit Cloud)
            # Call agents directly. In CLOUD_MODE=1 they will use OpenAI via langchain_openai.
            from src.agents.grammar_agent import GrammarAgent
            from src.agents.exam_agent import ExamAgent
            from src.agents.scorer_agent import ScorerAgent

            if mode == "fix":
                out = GrammarAgent().fix(text)
                st.code(out, language="markdown")
            elif mode == "evaluate":
                out = ExamAgent().evaluate(text)
                st.code(out, language="markdown")
            else:
                out = ScorerAgent().score(text)
                st.json(out)

        st.success("Done ✅")
    except Exception as e:
        st.error(f"Request failed: {safe_error(e)}")

# Footer info
with st.expander("⚙️ Runtime info"):
    st.write(
        {
            "API_BASE": API_BASE or "(none; using in-process agents)",
            "CLOUD_MODE": CLOUD_MODE,
            "CLOUD_MODEL": os.getenv("CLOUD_MODEL", ""),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
        }
    )
