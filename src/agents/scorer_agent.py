import os
from langchain.prompts import PromptTemplate
from src.utils.memory import memory
from src.prompts.persona import CORE_PERSONA
from src.llm.providers import build_client

RUBRIC = """You are an examiner grading Norwegian language output.
Assess it based on:
1. Grammar accuracy
2. Vocabulary range
3. Sentence structure
4. Coherence

Respond in JSON with:
{{
  "score": <float between 0 and 1>,
  "feedback": "<one short paragraph>"
}}
"""

def _json_recover(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"level": "A2", "score": 60, "rationale": "Could not parse model output."}

class ScorerAgent:
    def __init__(self, model: str | None = None):
        self.llm = build_client(task="scoring")
        self.tmpl = PromptTemplate.from_template(PROMPT)

    def score(self, text: str) -> dict:
        prompt = CORE_PERSONA + "\n\n" + self.tmpl.format(rubric=RUBRIC, text=text)
        raw = self.llm.predict(prompt).strip()
        data = _json_recover(raw)

        # normalize & validate
        level = str(data.get("level", "A2")).upper()
        if level not in {"A1", "A2", "B1", "B2"}:
            level = "A2"
        try:
            score = int(data.get("score", 60))
        except Exception:
            score = 60
        rationale = str(data.get("rationale", "")).strip()

        # optional subscores passthrough if your prompt/LLM returns them
        grammar = data.get("grammar")
        logic   = data.get("logic")
        vocab   = data.get("vocab")

        out = {"level": level, "score": score, "rationale": rationale}
        if grammar is not None: out["grammar"] = grammar
        if logic   is not None: out["logic"]   = logic
        if vocab   is not None: out["vocab"]   = vocab
        return out