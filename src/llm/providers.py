# src/llm/providers.py

import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Load .env so local variables (like OLLAMA_MODEL) are available even in one-off scripts
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass



# --- Base interface ---
class LLMClient:
    """A simple unified interface for all language model providers."""
    def predict(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

# --- OpenAI-compatible (OpenAI, OpenRouter, Perplexity via base_url) ---
class OpenAICompatClient(LLMClient):
    """Handles OpenAI and any other API that uses the OpenAI format."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True
    )
    def predict(self, prompt: str, **kwargs) -> str:
        """Send a text prompt and return model output."""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return r.choices[0].message.content.strip()


# --- Gemini (Google Generative AI) ---
class GeminiClient(LLMClient):
    """Handles requests to Gemini models using google-generativeai."""
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True
    )
    def predict(self, prompt: str, **kwargs) -> str:
        """Send a text prompt and return model output."""
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        return (response.text or "").strip()


# --- Ollama (local fallback for offline use) ---
class OllamaClient(LLMClient):
    """Uses a locally running Ollama model for offline inference."""
    def __init__(self, model: str, temperature: float = 0.2):
        from langchain_ollama import OllamaLLM
        self.llm = OllamaLLM(model=model, temperature=temperature, num_ctx=2048)

    def predict(self, prompt: str, **kwargs) -> str:
        return self.llm.predict(prompt).strip()

# --- Router ---
def build_client(task: str = "general") -> LLMClient:
    """
    Selects the appropriate LLM provider based on available environment variables.
    task: 'grammar' | 'reasoning' | 'scoring' | 'general'
    """
    routing = os.getenv("LLM_ROUTING", "auto").lower()

    # 1) Perplexity for reasoning if configured
    if task == "reasoning" and os.getenv("PPLX_API_KEY"):
        print(f"🧠 [Router] Task={task} → Provider=Perplexity ({os.getenv('PPLX_MODEL_REASON')})")
        return OpenAICompatClient(
            api_key=os.getenv("PPLX_API_KEY"),
            model=os.getenv("PPLX_MODEL_REASON", "llama-3.1-sonar-large-128k-online"),
            base_url=os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
            temperature=0.2,
        )

    # 2) OpenAI (default for most other tasks)
    if os.getenv("OPENAI_API_KEY"):
        print(f"⚡ [Router] Task={task} → Provider=OpenAI ({os.getenv('CLOUD_MODEL', 'gpt-4o-mini')})")
        model = os.getenv("CLOUD_MODEL") or os.getenv("OPENAI_MODEL_CHEAP", "gpt-4o-mini")
        return OpenAICompatClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.2,
        )

    # 3) Gemini fallback
    if os.getenv("GEMINI_API_KEY"):
        print(f"🌸 [Router] Task={task} → Provider=Gemini ({os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')})")
        return GeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            temperature=0.2,
        )

    # 4) Ollama local fallback
    if os.getenv("OLLAMA_MODEL"):
        print(f"💻 [Router] Task={task} → Provider=Ollama ({os.getenv('OLLAMA_MODEL')})")
        return OllamaClient(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            temperature=0.2,
        )

    raise RuntimeError("No valid LLM provider configured. Check your environment variables.")
