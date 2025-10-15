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
    """Exponential moving average for visibly responsive meters."""
    try:
        x = float(x)
    except Exception:
        x = 0.0
    if prev is None:
        return round(x, 2)
    return round(alpha * x + (1 - alpha) * float(prev), 2)

def map_cefr(total):
    """Simple CEFR map from total score; expand to B2 later."""
    if total is None:
        return "—"
    t = float(total)
    if t < 45:
        return "A1"
    if t < 70:
        return "A2"
    return "B1"

def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.ema = {"grammar": None, "logic": None, "vocab": None, "total": None}
    st.session_state.turns = 0
    st.session_state.user_profile = {"predicted_cefr": None, "ema_total": None}

def next_question_from_topic(user_text: str, level: str) -> str:
    """Generate ONE short follow-up question to keep the conversation flowing."""
    level = level or "A2"
    seed = f"""You are a Norwegian oral examiner. Based on the user's last message:

User: {user_text}

Write ONE short follow-up question in Norwegian at {level} level (A1/A2/B1). Keep it natural and under 120 characters.
Return only the question."""
    llm = build_client("reasoning")
    out = llm.predict(seed).strip()
    # Keep it to one line
    return out.split("\n")[0][:200]


# ---------------- Header: compact meters + controls ----------------
col_a, col_b = st.columns([3, 1])
with col_a:
    total = st.session_state.ema["total"] or 0
    st.progress(int(total))
    sub_a, sub_b, sub_c, sub_d = st.columns(4)
    sub_a.metric("Grammar", st.session_state.ema["grammar"] or 0)
    sub_b.metric("Logic",   st.session_state.ema["logic"] or 0)
    sub_c.metric("Vocab",   st.session_state.ema["vocab"] or 0)
    sub_d.metric("CEFR",    map_cefr(st.session_state.ema["total"]))
with col_b:
    if st.button("🔄 Ny økt"):
        reset_session()
        st.rerun()

# Small level badge (predicted CEFR) under title
lvl = st.session_state.user_profile["predicted_cefr"] or "—"
st.markdown(f"<div style='text-align:right'>Nivå (pred.): <b>{lvl}</b></div>", unsafe_allow_html=True)

# Grade conversation button (formal examiner report)
grade_now = st.button("📄 Vurder samtale (Norskprøven-stil)")


# ---------------- Grade (examiner report) ----------------
if grade_now and st.session_state.messages:
    history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages)
    examiner_prompt = f"""
ROLE SWITCH: You are now an official Norskprøven Examiner. Tone: formal, authoritative, fact-based.

INPUT DATA (full interaction):
{history}

TASK: Analyze the entire interaction. Output a single, structured Markdown response.

MANDATORY OUTPUT STRUCTURE:

### Norskprøven Examination Report
**1. Estimated CEFR Level:** [A1, A2, B1, or B2]
**2. Overall Justification:** [One detailed paragraph explaining why, referencing complexity, range, control.]
**3. Top 3 Errors Identified:**
* [Error 1]
* [Error 2]
* [Error 3]
**4. Actionable Next Steps:** [Three concrete, focused study recommendations.]
""".strip()
    llm = build_client("scoring")
    report = llm.predict(examiner_prompt).strip()
    st.session_state.messages.append({"role": "assistant", "content": report})
    with st.chat_message("assistant"):
        st.markdown(report)
    st.stop()


# ---------------- Chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- Chat input + turn handling ----------------
user_text = st.chat_input("Skriv eller snakk norsk her …")
if user_text:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Run agents
    try:
        sid = st.session_state.session_id

        # a) Grammar correction (short)
        grammar_out = GrammarAgent().fix(user_text, session_id=sid)

        # b) Exam-style evaluation/tip
        eval_out = ExamAgent().evaluate(user_text, session_id=sid)

        # c) Scoring (expects at least {"level": "...", "score": ...})
        score = ScorerAgent().score(user_text)

        # Update EMA meters
        total_score = int(score.get("score", 60))
        grammar_score = int(score.get("grammar", total_score))
        logic_score   = int(score.get("logic",   total_score))
        vocab_score   = int(score.get("vocab",   total_score))

        st.session_state.ema["grammar"] = ema_update(st.session_state.ema["grammar"], grammar_score)
        st.session_state.ema["logic"]   = ema_update(st.session_state.ema["logic"],   logic_score)
        st.session_state.ema["vocab"]   = ema_update(st.session_state.ema["vocab"],   vocab_score)
        st.session_state.ema["total"]   = ema_update(st.session_state.ema["total"],   total_score)
        st.session_state.turns += 1

        # Silent profile update
        st.session_state.user_profile["predicted_cefr"] = score.get("level", map_cefr(st.session_state.ema["total"]))
        st.session_state.user_profile["ema_total"] = st.session_state.ema["total"]

        # Follow-up question to continue the conversation
        follow = next_question_from_topic(user_text, st.session_state.user_profile.get("predicted_cefr"))

        # Compose assistant reply (loop: correction → explanation → continue)
        reply = f"""**🔧 Correction**
{grammar_out}

**🧪 Evaluation**
{eval_out}

**📊 Score**
Level: `{score.get('level','A2')}` • Total: `{total_score}`
_{score.get('rationale','')}_

**👉 Fortsettelse**
{follow}
"""

        # Append + render assistant
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        st.toast("Oppdatert score!", icon="✅")

    except Exception as e:
        err = f"Beklager, noe gikk galt. ({e})"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
