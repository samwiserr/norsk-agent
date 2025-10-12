from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

RUBRIC = """
You are a CEFR assessor for Norwegian (A1–B1). Given a single learner sentence:
1) Assign a CEFR level from {levels} (pick ONE).
2) Score 0–100.
3) Justify briefly in English (<= 4 lines).
Only return JSON with keys: level, score, rationale.
"""

PROMPT = """{rubric}

Sentence:
{text}

Return JSON only.
"""

class ScorerAgent:
    def __init__(self, model: str = "mistral"):
        self.llm = Ollama(model=model)
        self.tmpl = PromptTemplate.from_template(PROMPT)

    def score(self, text: str) -> dict:
        prompt = self.tmpl.format(rubric=RUBRIC.format(levels="A1,A2,B1"), text=text)
        raw = self.llm.invoke(prompt).strip()
        # very simple JSON recovery
        import json, re
        try:
            js = json.loads(raw)
        except Exception:
            js = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return {
            "level": js.get("level", "A2"),
            "score": int(js.get("score", 60)),
            "rationale": js.get("rationale", "")
        }
