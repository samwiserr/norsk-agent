# src/agents/exam_agent.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

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

    def evaluate(self, text: str) -> str:
        prompt = self.prompt.format(system=SYSTEM_INSTRUCTIONS, text=text)
        return self.llm.invoke(prompt).strip()

