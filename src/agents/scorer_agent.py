# src/agents/scorer_agent.py
import json
import re
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from src.llm.providers import build_client
from src.prompts.persona import CORE_PERSONA  # ok to keep for consistency

# ---- Prompt & Rubric ----
PROMPT = """You are an official Bokmål Norskprøven examiner. Grade the student's SINGLE sentence (or short turn)
and return STRICT JSON ONLY with these keys:

{{
  "level": "A1" | "A2" | "B1" | "B2",
  "grammar": <integer 0..100>,
  "logic": <integer 0..100>,
  "vocab": <integer 0..100>,
  "score": <integer 0..100>,  // overall
  "rationale": "<max 3 concise sentences>"
}}

Rules:
- "grammar" = grammatical accuracy (morphology, agreement, word order, tense)
- "logic"   = coherence/structure (clear meaning, cohesion, connectors)
- "vocab"   = lexical range & appropriateness (word choice, collocations, spelling)
- Compute overall "score" as: round(0.45*grammar + 0.25*logic + 0.30*vocab)

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

def _json_recover(raw: str) -> Dict[str, Any]:
    """Try hard to parse a dict out of possibly messy model output."""
    raw = (raw or "").strip()

    # direct parse if starts with {
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            pass

    # otherwise extract first {...} block
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # fallback
    return {"level": "A2", "score": 60, "rationale": "Parser fallback: non-JSON model output."}

class ScorerAgent:
    def __init__(self, llm=None, model: str | None = None):
        # allow injected client; fallback to router
        self.llm = llm or build_client(task="scoring")
        self.tmpl = PromptTemplate.from_template(PROMPT)

    def score(self, text: str) -> Dict[str, Any]:
        prompt = CORE_PERSONA + "\n\n" + RUBRIC + "\n\n" + self.tmpl.format(text=text)
        raw = self.llm.predict(prompt).strip()
        data = _json_recover(raw)

        level = str(data.get("level", "A2")).upper().strip()
        if level not in {"A1", "A2", "B1", "B2"}:
            level = "A2"

        def _clamp_int(v, default=60):
            try:
                v = int(v)
            except Exception:
                v = default
            return max(0, min(100, v))

        grammar = _clamp_int(data.get("grammar"), 60)
        logic   = _clamp_int(data.get("logic"),   60)
        vocab   = _clamp_int(data.get("vocab"),   60)

        score = data.get("score")
        if score is None:
            score = round(0.45*grammar + 0.25*logic + 0.30*vocab)
        score = _clamp_int(score, 60)

        rationale = str(data.get("rationale", "")).strip()

        return {
            "level": level,
            "grammar": grammar,
            "logic": logic,
            "vocab": vocab,
            "score": score,
            "rationale": rationale,
        }
