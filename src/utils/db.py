# src/utils/db.py
import os, time, sqlite3, json
from pathlib import Path

# Project root = two levels up from this file (src/utils/db.py)
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = str(BASE_DIR / "norskagent.db")


def _ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,
      mode TEXT NOT NULL,
      session_id TEXT,
      input TEXT NOT NULL,
      output TEXT NOT NULL,
      meta TEXT
    );
    """)
    conn.commit()

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    return conn

def log_interaction(mode: str, session_id: str | None, input_text: str, output_text: str, meta: dict | None = None):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO logs (ts, mode, session_id, input, output, meta) VALUES (?,?,?,?,?,?)",
            (time.time(), mode, session_id, input_text, output_text, json.dumps(meta or {}))
        )
        conn.commit()
    finally:
        conn.close()
