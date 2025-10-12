# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.agents.exam_agent import ExamAgent
from src.agents.grammar_agent import GrammarAgent
from fastapi.middleware.cors import CORSMiddleware

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
def evaluate(input: TextIn):
    """
    Exam-style evaluation:
    - Corrected sentence
    - Explanation (English)
    - One tip
    """
    result = exam_agent.evaluate(input.text)
    return {"mode": "evaluate", "input": input.text, "result": result}

@app.post("/fix")
def fix(input: TextIn):
    """
    Grammar fix:
    - Corrected sentence
    - Brief explanation (English)
    """
    result = grammar_agent.fix(input.text)
    return {"mode": "fix", "input": input.text, "result": result}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)
