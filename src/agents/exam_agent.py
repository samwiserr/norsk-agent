from src.llm.providers import build_client
from langchain.prompts import PromptTemplate
from src.utils.memory import memory
from src.prompts.persona import CORE_PERSONA

SYSTEM_INSTRUCTIONS = (
    "You are a Norwegian Bokmål language evaluator (A1–B1) who reviews one sentence at a time.\n"
    "Rules:\n"
    "- Return exactly two short parts: Explanation (in English), Tip (in English).\n"
    "- Use Norwegian letters (æ, ø, å) in examples.\n"
    "- Keep the explanation simple (maximum 2–3 lines).\n"
    "- Do NOT continue the dialogue.\n"
)

EVAL_TEMPLATE = """{system}

Example:
Input: "Jer er trott"
Output:
“Jer” → “Jeg”; “trott” → “trøtt” (ø). The verb “er” is already correct in present tense.
Tip: Practice the vowels æ/ø/å in common adjectives (trøtt = tired, blå = blue, små = small).

Use the same format for the user’s sentence below.

User sentence:
{text}

"""

class ExamAgent:
    def __init__(self, model: str | None = None):
        self.llm = build_client(task="reasoning")
        self.prompt = PromptTemplate.from_template(EVAL_TEMPLATE)

    def evaluate(self, text: str, session_id: str | None = None) -> str:
        history = memory.get(session_id)
        # Persona + your evaluation prompt
        prompt = CORE_PERSONA + "\n\n" + self.prompt.format(
            system=SYSTEM_INSTRUCTIONS, text=text
        )

        if history:
            prompt += "\n\nKontekst (tidligere meldinger):\n" + "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history
            )

        out = self.llm.predict(prompt).strip()
        memory.append(session_id, "user", text)
        memory.append(session_id, "assistant", out)
        return out
