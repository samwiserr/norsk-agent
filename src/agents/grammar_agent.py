import os
from langchain.prompts import PromptTemplate
from src.utils.memory import memory

# ===== Cloud / Local switch =====
CLOUD_MODE = os.getenv("CLOUD_MODE", "0") == "1"

if CLOUD_MODE:
    # Hosted LLM (OpenAI-compatible)
    from langchain_openai import ChatOpenAI

    def make_llm():
        return ChatOpenAI(
            model=os.getenv("CLOUD_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", None),  # optional for OpenAI
            temperature=0.2,
        )
else:
    # Local Ollama
    from langchain_ollama import OllamaLLM

    def make_llm():
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            temperature=0.2,
            num_ctx=2048,
        )

# ===== Your tuned system + few-shot template =====
SYSTEM_INSTRUCTIONS = (
    "Du er en hjelpsom norsk grammatikkassistent.\n"
    "Målgruppe: A1–B1.\n"
    "Regler:\n"
    "- Svar kort og presist.\n"
    "- Bruk riktige norske diakritiske tegn: æ, ø, å.\n"
    "- Behold person og tid hvis ikke setningen krever endring.\n"
    "- Svar på norsk, men forklar grammatikken kort på enkel engelsk.\n"
)

FIX_TEMPLATE = """{system}

Eksempel:
Input: "Jer er trott"
Output:
Corrected: Jeg er trøtt.
Explanation: “Jer” → “Jeg”; “trott” → “trøtt” (ø). “er” korrekt i presens.

Bruk samme format for brukerens setning under.

User sentence:
{text}

Respond in this format:
Corrected:
Explanation:
"""

class GrammarAgent:
    def __init__(self, model: str | None = None):
        self.llm = make_llm()
        self.prompt = PromptTemplate.from_template(FIX_TEMPLATE)

    def fix(self, text: str, session_id: str | None = None) -> str:
        history = memory.get(session_id)
        prompt = self.prompt.format(system=SYSTEM_INSTRUCTIONS, text=text)
        if history:
            history_block = "\n\nKontekst (tidligere meldinger):\n" + "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history
            )
            prompt += history_block
        out = self.llm.predict(prompt).strip()
        memory.append(session_id, "user", text)
        memory.append(session_id, "assistant", out)
        return out