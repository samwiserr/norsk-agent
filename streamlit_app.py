# streamlit_app.py
import os
import uuid
import streamlit as st

from src.agents.grammar_agent import GrammarAgent
from src.agents.exam_agent import ExamAgent
from src.agents.scorer_agent import ScorerAgent
from src.llm.providers import build_client


# ---------- Page setup ----------
st.set_page_config(page_title="Norsk Agent", page_icon="🇳🇴", layout="centered")
st.title("🇳🇴 Norsk Agent")
st.caption("A1–B1 norsktrening: grammatikk, eksamensstil tilbakemelding og CEFR-scoring.")

# Session ID for memory (persists during a Streamlit session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

col_left, col_right = st.columns([1, 1])
with col_left:
    if st.button("🔄 Start ny økt (reset minne)"):
        st.session_state.session_id = str(uuid.uuid4())
        st.success("Ny økt startet. Minne er nullstilt.")


# ---------- Input UI ----------
text = st.text_area("Skriv en setning på norsk:", height=140, placeholder="F.eks. Jer er trott")
mode = st.selectbox("Velg modus", ["fix", "evaluate", "score"], index=0)
go = st.button("Kjør")


# ---------- Runtime panel (sidebar) ----------
def _active_info(task: str):
    """Return provider, model and endpoint chosen by the router for a given task."""
    try:
        c = build_client(task)
        t = type(c).__name__
        if t == "OpenAICompatClient":
            # If reasoning and Perplexity key is present, router uses Perplexity via OpenAI-compat client.
            if task == "reasoning" and os.getenv("PPLX_API_KEY"):
                return {
                    "task": task, "provider": "Perplexity",
                    "model": os.getenv("PPLX_MODEL_REASON", "llama-3.1-sonar-large-128k-online"),
                    "endpoint": os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
                }
            # Otherwise OpenAI
            return {
                "task": task, "provider": "OpenAI",
                "model": os.getenv("CLOUD_MODEL", os.getenv("OPENAI_MODEL_CHEAP", "gpt-4o-mini")),
                "endpoint": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            }
        if t == "GeminiClient":
            return {
                "task": task, "provider": "Gemini",
                "model": os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                "endpoint": "google-generativeai",
            }
        if t == "OllamaClient":
            return {
                "task": task, "provider": "Ollama",
                "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                "endpoint": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            }
        return {"task": task, "provider": t, "model": "?", "endpoint": "?"}
    except Exception as e:
        return {"task": task, "provider": "ERROR", "model": str(e), "endpoint": ""}


def render_runtime_panel():
    st.sidebar.header("⚙️ Runtime")
    st.sidebar.write({"session_id": st.session_state.session_id})
    for task in ("grammar", "reasoning", "scoring"):
        info = _active_info(task)
        st.sidebar.markdown(
            f"**{task.title()}** → {info['provider']}\n\n"
            f"- model: `{info['model']}`\n"
            f"- endpoint: `{info['endpoint']}`"
        )
    st.sidebar.divider()
    st.sidebar.write({
        "CLOUD_MODE": os.getenv("CLOUD_MODE"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "CLOUD_MODEL": os.getenv("CLOUD_MODEL"),
        "Perplexity?": bool(os.getenv("PPLX_API_KEY")),
        "Gemini?": bool(os.getenv("GEMINI_API_KEY")),
    })


# ---------- Run selected task ----------
if go and text.strip():
    try:
        if mode == "fix":
            out = GrammarAgent().fix(text, session_id=st.session_state.session_id)
            st.subheader("🔧 Korreksjon")
            st.code(out, language="markdown")

        elif mode == "evaluate":
            out = ExamAgent().evaluate(text, session_id=st.session_state.session_id)
            st.subheader("🧪 Vurdering")
            st.code(out, language="markdown")

        else:  # score
            data = ScorerAgent().score(text)
            st.subheader("📊 CEFR-score")
            st.json(data)

        st.success("Done ✅")
    except Exception as e:
        st.error(f"Request failed: {e}")

# Render sidebar info at the end so it reflects any UI overrides (e.g., model choice)
#render_runtime_panel()
