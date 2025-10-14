# src/agents/scorer_agent.py
import json
import re
from langchain.prompts import PromptTemplate
from src.llm.providers import build_client

RUBRIC = (
    "You are a CEFR assessor for Norwegian (A1–B1). Given ONE learner sentence, do three things:\n"
    "1) Assign a CEFR level from [A1, A2, B1] (pick ONE).\n"
    "2) Give a numeric score 0–100.\n"
    "3) Justify briefly in English (max 4 lines).\n"
    "IMPORTANT: Return STRICT JSON with keys exactly: level, score, rationale."
)

PROMPT = """{rubric}

Example input:
"Jer er trott"

Expected JSON example:
{{"level":"A1","score":40,"rationale":"Misspelling of 'Jeg' and 'trøtt'; basic present tense is otherwise fine."}}

Sentence:
{text}

Return JSON ONLY.
"""

class ScorerAgent:
    def __init__(self, model: str | None = None):
        # Provider is chosen centrally (OpenAI / Gemini / Ollama)
        self.llm = build_client(task="scoring")
        self.tmpl = PromptTemplate.from_template(PROMPT)

    def _json_recover(self, raw: str) -> dict:
        # try strict json first
        try:
            return json.loads(raw)
        except Exception:
            pass
        # try to extract first {...} block
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # fallback
        return {"level": "A2", "score": 60, "rationale": "Could not parse model output."}

    def score(self, text: str) -> dict:
        prompt = self.tmpl.format(rubric=RUBRIC, text=text)
        raw = self.llm.predict(prompt).strip()
        data = self._json_recover(raw)

        level = str(data.get("level", "A2")).upper()
        if level not in {"A1", "A2", "B1"}:
            level = "A2"

        try:
            score = int(data.get("score", 60))
        except Exception:
            score = 60

        rationale = str(data.get("rationale", "")).strip()
        return {"level": level, "score": score, "rationale": rationale}
