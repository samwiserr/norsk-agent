import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print("OLLAMA_HOST =", base)
llm = OllamaLLM(model="mistral", base_url=base)
prompt = PromptTemplate.from_template("Correct this Norwegian sentence:\n{text}\nReturn only the corrected sentence.")
out = llm.invoke(prompt.format(text="Jer er trott"))
print("OK ->", out[:200])
