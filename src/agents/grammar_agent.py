# src/agents/grammar_agent.py
from src.llm.providers import build_client
from langchain.prompts import PromptTemplate
from src.utils.memory import memory
from src.prompts.persona import CORE_PERSONA

SYSTEM_INSTRUCTIONS = (
    "Du er en norsk grammatikkassistent (A1–B1-nivå) som hjelper brukeren å korrigere setninger.\n"
    "Regler:\n"
    "- Returner tre korte deler: Corrected, Explanation (English), Tip.\n"
    "- Bruk norske bokstaver (æ, ø, å).\n"
    "- Forklar enkelt, maks 2–3 linjer.\n"
)

FIX_TEMPLATE = """{system}

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

class GrammarAgent:
    def __init__(self, model: str | None = None):
        # Centralized provider (OpenAI/Perplexity/Gemini/Ollama)
        self.llm = build_client(task="grammar")
        self.prompt = PromptTemplate.from_template(FIX_TEMPLATE)

    def fix(self, text: str, session_id: str | None = None) -> str:
        # Build prompt with persona + your template
        prompt = CORE_PERSONA + "\n\n" + self.prompt.format(
            system=SYSTEM_INSTRUCTIONS, text=text
        )

        # Include short-term memory context if present
        history = memory.get(session_id)
        if history:
            prompt += "\n\nKontekst (tidligere meldinger):\n" + "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history
            )

        # Call LLM
        out = self.llm.predict(prompt).strip()

        # Save to memory
        memory.append(session_id, "user", text)
        memory.append(session_id, "assistant", out)
        return out
