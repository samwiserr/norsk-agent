from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.exam_agent import ExamAgent
from src.agents.grammar_agent import GrammarAgent
from src.agents.scorer_agent import ScorerAgent
from src.utils.db import log_interaction

app = FastAPI(title="NorskAgent API")

exam_agent = ExamAgent()
grammar_agent = GrammarAgent()
scorer_agent = ScorerAgent()

class TextIn(BaseModel):
    text: str


# ---------- Endpoints ----------
@app.post("/evaluate")
def evaluate(input: TextIn, x_session_id: str | None = Header(default=None), dry_run: bool = False):
    try:
        if dry_run:
            result = "Corrected: Jeg er trøtt.\nExplanation: Demo forklaring.\nTip: Øv på ø-lyden."
        else:
            result = exam_agent.evaluate(input.text, session_id=x_session_id)

        log_interaction("evaluate", x_session_id, input.text, result)
        return {
            "mode": "evaluate",
            "session_id": x_session_id,
            "input": input.text,
            "result": result
        }
    except Exception as e:
        # surfaces the actual issue instead of a bare 500
        raise HTTPException(status_code=503, detail=f"ExamAgent error: {e}")


@app.post("/fix")
def fix(input: TextIn, x_session_id: str | None = Header(default=None)):
    try:
        result = grammar_agent.fix(input.text, session_id=x_session_id)
        log_interaction("fix", x_session_id, input.text, result)
        return {"mode": "fix", "session_id": x_session_id, "input": input.text, "result": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"GrammarAgent error: {e}")

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
def score(input: TextIn, x_session_id: str | None = Header(default=None), dry_run: bool = False):
    try:
        if dry_run:
            payload = {"level": "A2", "score": 62, "rationale": "Demo: basic tense error; simple vocab correct."}
        else:
            payload = scorer_agent.score(input.text)  # returns dict {level, score, rationale}

        # compact string for quick table views; keep full dict in meta
        compact = f"Level: {payload['level']} | Score: {payload['score']}"
        log_interaction("score", x_session_id, input.text, compact, meta=payload)

        return {
            "mode": "score",
            "session_id": x_session_id,
            "input": input.text,
            **payload
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ScorerAgent error: {e}")

@app.get("/health")
def health():
    import os, requests
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "")
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        r.raise_for_status()
        return {"ok": True, "ollama": "reachable", "base": base, "model": model}
    except Exception as e:
        return {"ok": False, "ollama": f"unreachable: {e}", "base": base, "model": model}

    