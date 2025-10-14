import os
import streamlit as st
from src.agents.grammar_agent import GrammarAgent
from src.agents.exam_agent import ExamAgent
from src.agents.scorer_agent import ScorerAgent

st.set_page_config(page_title="Norsk Agent", page_icon="🇳🇴", layout="centered")
st.title("🇳🇴 Norsk Agent")

st.caption("A1–B1 norsktrening: grammatikk, eksamensstil tilbakemelding og CEFR-scoring.")

text = st.text_area("Skriv en setning på norsk:", height=120, placeholder="F.eks. Jer er trott")
mode = st.selectbox("Velg modus", ["fix", "evaluate", "score"], index=0)

if st.button("Kjør") and text.strip():
    try:
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
        st.error(f"Request failed: {e}")

with st.expander("⚙️ Runtime info"):
    st.write({
        "CLOUD_MODE": os.getenv("CLOUD_MODE"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "CLOUD_MODEL": os.getenv("CLOUD_MODEL"),
        "Perplexity?": bool(os.getenv("PPLX_API_KEY")),
        "Gemini?": bool(os.getenv("GEMINI_API_KEY")),
    })

if st.button("🔎 Test OpenAI"):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    r = client.chat.completions.create(model=os.getenv("OPENAI_MODEL_CHEAP","gpt-4o-mini"),
                                       messages=[{"role":"user","content":"Say OK only."}])
    st.write("OpenAI says:", r.choices[0].message.content)

