# src/agents/grammar_agent.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from src.utils.memory import memory

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

    def fix(self, text: str, session_id: str | None = None) -> str:
        history = []
        if session_id:
            history = memory.get(session_id)
        prompt = self.prompt.format(system=SYSTEM_INSTRUCTIONS, text=text)
        if history:
            history_block = "\n\nContext:\n" + "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history
            )
            prompt = prompt + history_block
        out = self.llm.invoke(prompt).strip()
        if session_id:
            memory.append(session_id, "user", text)
            memory.append(session_id, "assistant", out)
        return out
