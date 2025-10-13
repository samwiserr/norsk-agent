#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI in background
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground (so container stays alive)
streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
