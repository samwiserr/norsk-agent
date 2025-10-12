# NorskAgent 🇳🇴
An open-source AI-driven language exam tutor built with Agentic AI.

## Overview
NorskAgent helps learners prepare for the official Norwegian language exams (A1–B1) using intelligent, interactive agents.

## Quickstart
1. Create virtualenv: `python -m venv venv`
2. Activate venv: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install deps: `pip install -r requirements.txt`
4. Set `OPENAI_API_KEY` environment variable.
5. Run: `uvicorn backend.app:app --reload`

## Project Structure
See `backend/` for the FastAPI app and agent code.
