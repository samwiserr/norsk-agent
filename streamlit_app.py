# streamlit_app.py
import os
import uuid
import streamlit as st

from src.agents.grammar_agent import GrammarAgent
from src.agents.exam_agent import ExamAgent
from src.agents.scorer_agent import ScorerAgent
from src.llm.providers import build_client


# ---------------- Page / session setup ----------------
st.set_page_config(page_title="Norsk Agent", page_icon="🇳🇴", layout="centered")
st.title("🇳🇴 Norsk Agent")
st.caption("Snakk norsk med en veileder. Få korrigering, forklaring og løpende CEFR-score (A1–B1).")

# Session primitives
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: "user"/"assistant", "content": "..."}]
if "ema" not in st.session_state:
    st.session_state.ema = {"grammar": None, "logic": None, "vocab": None, "total": None}
if "turns" not in st.session_state:
    st.session_state.turns = 0
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"predicted_cefr": None, "ema_total": None}


# ---------------- Helpers ----------------
def ema_update(prev, x, alpha=0.3):
    if prev is None:
        return x
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return round(alpha * x + (1 - alpha) * float(prev), 2)

def map_cefr(total):
    if total is None:
        return "—"
    t = float(total)
    if t < 45:
        return "A1"
    if t < 70:
        return "A2"
    return "B1"  # expand later

def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.ema = {"grammar": None, "logic": None, "vocab": None, "total": None}
    st.session_state.turns = 0
    st.session_state.user_profile = {"predicted_cefr": None, "ema_total": None}

def next_question_from_topic(user_text: str,