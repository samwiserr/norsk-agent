# src/agents/grammar_agent.py
from src.llm.providers import build_client
from langchain.prompts import PromptTemplate
from src.utils.memory import memory
from src.prompts.persona import CORE_PERSONA

SYSTEM_INSTRUCTIONS = (
    "You are a Norwegian grammar assistant (A1–B2).\n"
    "Return EXACTLY three parts and NOTHING else:\n"
    "1) Corrected: <single corrected Norwegian sentence>\n"
    "2) Explanation: <short explanation in English, max 2–3 lines>\n"
    "3) Tip: <one short practical tip in English>\n"
    "Rules:\n"
    "- Use Norwegian letters (æ, ø, å) in the corrected sentence.\n"
    "- Do NOT ask any question.\n"
    "- Do NOT continue the dialogue.\n"
)

FIX_TEMPLATE = """{system}

Example:
Input: "Han gå til butikk i går"
Output:
Corrected: Han gikk til butikken i går.
Explanation: “gå” (present tense) should be “gikk” (past tense). “butikk” needs the definite form “butikken” because of “til.”
Tip: Practice verb tenses (present vs. past) and definite forms of nouns.


Bruk samme format for brukerens setning under.

User sentence:
{text}

Respond in this format and nothing else:
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
