from src.llm.providers import build_client
from langchain.prompts import PromptTemplate
from src.utils.memory import memory
from src.prompts.persona import CORE_PERSONA

SYSTEM_INSTRUCTIONS = (
    "Du er en norsk språksensor (A1–B1) som vurderer én setning av gangen.\n"
    "Regler:\n"
    "- Returner tre korte deler: Corrected, Explanation (English), Tip.\n"
    "- Bruk norske diakritiske tegn (æ, ø, å).\n"
    "- Hold forklaringen enkel (maks 2–3 linjer).\n"
)

EVAL_TEMPLATE = """{system}

Eksempel:
Input: "Jer er trott"
Output:
Corrected: Jeg er trøtt.
Explanation: “Jer” → “Jeg”; “trott” → “trøtt” (ø). Presens “er” er riktig.
Tip: Øv på vokalene æ/ø/å i vanlige adjektiv (trøtt, blå, små).

Bruk samme format for brukerens setning under.

User sentence:
{text}

Respond in this format:
Corrected:
Explanation:
Tip:
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
