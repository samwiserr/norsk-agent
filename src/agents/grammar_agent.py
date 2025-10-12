# src/agents/grammar_agent.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

SYSTEM_INSTRUCTIONS = (
    "You are a helpful Norwegian grammar assistant.\n"
    "Given a user's Norwegian sentence, provide:\n"
    "1) Corrected sentence (one line)\n"
    "2) Brief explanation of the grammar issues in simple English (2–3 lines max)\n"
    "Keep your answer short."
)

FIX_TEMPLATE = """{system}

User sentence:
{text}

Respond in this format:
Corrected:
Explanation:
"""

class GrammarAgent:
    """Agent that corrects a Norwegian sentence and explains briefly (using a local Ollama model)."""

    def __init__(self, model: str = "mistral"):
        self.llm = Ollama(model=model)
        self.prompt = PromptTemplate.from_template(FIX_TEMPLATE)

    def fix(self, text: str) -> str:
        prompt = self.prompt.format(system=SYSTEM_INSTRUCTIONS, text=text)
        return self.llm.invoke(prompt).strip()
