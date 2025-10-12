# dashboard_app.py
import os, time, sqlite3, json
import pandas as pd
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(__file__), "norskagent.db")

st.set_page_config(page_title="NorskAgent Dashboard", page_icon="📊", layout="wide")
st.title("📊 NorskAgent Dashboard")

if not os.path.exists(DB_PATH):
    st.info("No database found yet. Interact with the API/UI first to create logs.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT id, ts, mode, session_id, input, output, meta FROM logs ORDER BY id DESC LIMIT 1000", conn
)
conn.close()

if df.empty:
    st.info("No logs yet. Use the app and come back.")
    st.stop()

# Format time
df["time"] = df["ts"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total interactions", len(df))
c2.metric("Evaluate calls", int((df["mode"] == "evaluate").sum()))
c3.metric("Grammar Fix calls", int((df["mode"] == "fix").sum()))
c4.metric("CEFR Score calls", int((df["mode"] == "score").sum()))

st.divider()

# Filters
left, right = st.columns([2, 1])
with left:
    st.subheader("Recent Interactions")
    st.dataframe(df[["time", "mode", "session_id", "input", "output"]], use_container_width=True, height=400)

with right:
    st.subheader("Mode Breakdown (counts)")
    st.bar_chart(df["mode"].value_counts())

# Optional: show last CEFR rationales
cefr = df[df["mode"] == "score"].copy()
if not cefr.empty:
    st.divider()
    st.subheader("Recent CEFR rationales")
    # parse meta JSON
    def get_meta(x):
        try:
            return json.loads(x)
        except Exception:
            return {}
    cefr["meta_json"] = cefr["meta"].apply(get_meta)
    cefr["level"] = cefr["meta_json"].apply(lambda m: m.get("level", ""))
    cefr["num"] = cefr["meta_json"].apply(lambda m: m.get("score", ""))
    cefr["rationale"] = cefr["meta_json"].apply(lambda m: m.get("rationale", ""))
    st.dataframe(cefr[["time", "session_id", "level", "num", "rationale", "input"]], use_container_width=True)
