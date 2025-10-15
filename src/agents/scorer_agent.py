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

class ScorerAgent:
    def __init__(self):
        self.llm = build_client()
        self.tmpl = PromptTemplate.from_template("{rubric}\n\nText:\n{text}\n\nJSON:")

    def score(self, text: str, session_id: str | None = None):
        # Include persona in the context
        prompt = CORE_PERSONA + "\n\n" + self.tmpl.format(rubric=RUBRIC, text=text)

        # Get result from LLM
        raw = self.llm.predict(prompt).strip()

        # Store in memory
        memory.append(session_id, "user", text)
        memory.append(session_id, "assistant", raw)

        return raw
