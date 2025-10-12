# src/agents/exam_agent.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from src.utils.memory import memory

SYSTEM_INSTRUCTIONS = (
    "You are a Norwegian language examiner (A1–B1).\n"
    "Task: Evaluate the user's Norwegian sentence.\n"
    "Return three parts:\n"
    "1) Corrected sentence\n"
    "2) Explanation of grammar mistakes in simple English\n"
    "3) One short tip for improvement\n"
    "Keep it concise."
)

EVAL_TEMPLATE = """{system}

User sentence:
{text}

Respond in this format:
Corrected:
Explanation:
Tip:
"""

class ExamAgent:
    def __init__(self, model: str = "mistral"):
        self.llm = Ollama(model=model)
        self.prompt = PromptTemplate.from_template(EVAL_TEMPLATE)

    def evaluate(self, text: str, session_id: str | None = None) -> str:
        history = []
        if session_id:
            history = memory.get(session_id)
        prompt = self.prompt.format(system=SYSTEM_INSTRUCTIONS, text=text)
        if history:
            history_block = "\n\nContext:\n" + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
            prompt = prompt + history_block
        out = self.llm.invoke(prompt).strip()
        if session_id:
            memory.append(session_id, "user", text)
            memory.append(session_id, "assistant", out)
        return out

