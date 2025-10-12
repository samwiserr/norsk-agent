# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.agents.exam_agent import ExamAgent
from src.agents.grammar_agent import GrammarAgent
from fastapi.middleware.cors import CORSMiddleware
from src.agents.scorer_agent import ScorerAgent
from fastapi import Header

# ---------- App setup ----------
app = FastAPI(
    title="NorskAgent API",
    description="Local agent endpoints powered by Ollama",
    version="0.1.0",
)

# Create one instance of each agent to reuse (faster than re-creating each request)
exam_agent = ExamAgent(model="mistral")
grammar_agent = GrammarAgent(model="mistral")

# ---------- Request models ----------
class TextIn(BaseModel):
    text: str

# ---------- Endpoints ----------
@app.post("/evaluate")
def evaluate(input: TextIn, x_session_id: str | None = Header(default=None)):
    result = exam_agent.evaluate(input.text, session_id=x_session_id)
    return {
        "mode": "evaluate",
        "session_id": x_session_id,
        "input": input.text,
        "result": result
    }

@app.post("/fix")
def fix(input: TextIn, x_session_id: str | None = Header(default=None)):
    result = grammar_agent.fix(input.text, session_id=x_session_id)
    return {
        "mode": "fix",
        "session_id": x_session_id,
        "input": input.text,
        "result": result
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer_agent = ScorerAgent(model="mistral")

class TextIn(BaseModel):
    text: str

@app.post("/score")
def score(input: TextIn):
    result = scorer_agent.score(input.text)
    return {"mode": "score", "input": input.text, **result}

