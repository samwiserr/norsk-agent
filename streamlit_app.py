# streamlit_app.py
import os
import uuid
import streamlit as st

from src.agents.grammar_agent import GrammarAgent
from src.agents.exam_agent import ExamAgent
from src.agents.scorer_agent import ScorerAgent


# ---------------- Page / session setup ----------------
st.set_page_config(page_title="Norsk Agent", page_icon="🇳🇴", layout="centered")
st.title("🇳🇴 Norsk Agent")
st.caption("Snakk norsk med en veileder. Få korrigering, forklaring og løpende CEFR-score (A1–B1).")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: "user"/"assistant", "content": "..."}]

# rolling scores (EMA)
if "ema" not in st.session_state:
    st.session_state.ema = {"grammar": None, "logic": None, "vocab": None, "total": None}
if "turns" not in st.session_state:
    st.session_state.turns = 0


# ---------------- Helpers ----------------
def ema_update(prev, x, alpha=0.3):
    if prev is None:
        return x
    return round(alpha * x + (1 - alpha) * prev, 2)

def map_cefr(total):
    # simple thresholds; tweak later
    if total is None: return "—"
    if total < 45: return "A1"
    if total < 70: return "A2"
    return "B1"

def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.ema = {"grammar": None, "logic": None, "vocab": None, "total": None}
    st.session_state.turns = 0


# ---------------- Sidebar: live skill meter ----------------
with st.sidebar:
    st.subheader("📊 Fremdrift")
    total = st.session_state.ema["total"] or 0
    st.progress(int(total))  # 0–100
    st.write(f"**Nivå (CEFR):** {map_cefr(st.session_state.ema['total'])}")

    cols = st.columns(3)
    cols[0].metric("Grammar", st.session_state.ema["grammar"] or 0)
    cols[1].metric("Logic", st.session_state.ema["logic"] or 0)
    cols[2].metric("Vocab", st.session_state.ema["vocab"] or 0)

    st.caption(f"Økt: `{st.session_state.session_id[:8]}`  •  Tur: {st.session_state.turns}")
    if st.button("🔄 Start ny samtale (nullstill)"):
        reset_session()
        st.rerun()


# ---------------- Chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat input ----------------
user_text = st.chat_input("Skriv eller lim inn norsk her …")
if user_text:
    # 1) Show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) Run agents
    try:
        sid = st.session_state.session_id
        # a) grammar correction (short)
        grammar_out = GrammarAgent().fix(user_text, session_id=sid)

        # b) exam-style explanation/tip
        eval_out = ExamAgent().evaluate(user_text, session_id=sid)

        # c) scoring
        score = ScorerAgent().score(user_text)
        # Expect: {"level":"A2","score": 62,"rationale":"..."} or your richer version

        # 3) Update EMA meters (if you later expand ScorerAgent to return grammar/logic/vocab separately, use those)
        # For now, use total as 'score' and approximate subscores evenly if missing.
        total_score = int(score.get("score", 60))
        grammar_score = score.get("grammar", total_score)
        logic_score = score.get("logic", total_score)
        vocab_score = score.get("vocab", total_score)

        st.session_state.ema["grammar"] = ema_update(st.session_state.ema["grammar"], int(grammar_score))
        st.session_state.ema["logic"]   = ema_update(st.session_state.ema["logic"],   int(logic_score))
        st.session_state.ema["vocab"]   = ema_update(st.session_state.ema["vocab"],   int(vocab_score))
        st.session_state.ema["total"]   = ema_update(st.session_state.ema["total"],   total_score)
        st.session_state.turns += 1

        # 4) Compose assistant reply (tight and friendly)
        reply = f"""**🔧 Correction**
{grammar_out}

**🧪 Evaluation**
{eval_out}

**📊 Score**
Level: `{score.get('level','A2')}` • Total: `{total_score}`
_{score.get('rationale','')}_"""

        st.session_state.messages.append({"role": "assistant", "content": reply})

        # 5) Render assistant message
        with st.chat_message("assistant"):
            st.markdown(reply)

        st.toast("Oppdatert score!", icon="✅")

    except Exception as e:
        err = f"Beklager, noe gikk galt. ({e})"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
