# src/agents/scorer_agent.py
import json
import re
from typing import Dict, Any

from langchain.prompts import PromptTemplate
from src.llm.providers import build_client
from src.prompts.persona import CORE_PERSONA  # ok if you want a consistent persona; safe to keep

# ---- Prompt & Rubric ----
PROMPT = """You are an official Norskprøven examiner. Grade the student's SINGLE sentence (or short turn)
and return STRICT JSON ONLY with these keys:

{
  "level": "A1" | "A2" | "B1" | "B2",
  "score": <integer 0..100>,
  "rationale": "<max 3 concise sentences>"
}

Do not include any extra text, code fences, or commentary—return JSON ONLY.

Student text:
{text}
"""

RUBRIC = """CEFR rubric (Norwegian):
A1: Simple phrases/sentences. Many errors; meaning mostly clear.
A2: Simple connected sentences; limited vocab; frequent grammar issues.
B1: Handles familiar topics; some cohesion; moderate accuracy.
B2: More complex ideas; good control; wider vocabulary; few errors.
"""

# ---- Robust JSON recovery ----
def _json_recover(raw: str) -> Dict[str, Any]:
    """Try hard to get a dict out of a possibly messy model response."""
    raw = (raw or "").strip()

    # Fast path: starts with { and parses
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            pass

    # Otherwise, extract first {...} block
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Nothing parseable -> safe fallback
    return {"level": "A2", "score": 60, "rationale": "Parser fallback: non-JSON model output."}


class ScorerAgent:
    def __init__(self, model: str | None = None):
        # Route via your provider system (OpenAI / Perplexity / Gemini / Ollama)
        self.llm = build_client(task="scoring")
        # Keep the rubric in the template so model sees criteria
        self.tmpl = PromptTemplate.from_template(PROMPT)

    def score(self, text: str) -> Dict[str, Any]:
        # Build prompt with persona + rubric + user text
        prompt = CORE_PERSONA + "\n\n" + RUBRIC + "\n\n" + self.tmpl.format(text=text)

        # Get raw model output (may be messy)
        raw = self.llm.predict(prompt).strip()

        # Recover JSON
        data = _json_recover(raw)

        # Normalize fields
        level = str(data.get("level", "A2")).upper().strip()
        if level not in {"A1", "A2", "B1", "B2"}:
            level = "A2"

        try:
            score = int(data.get("score", 60))
        except Exception:
            score = 60
        score = max(0, min(100, score))

        rationale = str(data.get("rationale", "")).strip()

        # Optional subscores passthrough (if your prompt later returns them)
        out: Dict[str, Any] = {"level": level, "score": score, "rationale": rationale}
        for k in ("grammar", "logic", "vocab"):
            if k in data:
                out[k] = data[k]
        return out
