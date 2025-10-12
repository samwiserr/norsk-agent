from fastapi import FastAPI
from pydantic import BaseModel
from agents.grammar_agent import GrammarAgent

app = FastAPI(title="NorskAgent API")
grammar_agent = GrammarAgent()

class TextInput(BaseModel):
    text: str

@app.post("/grammar")
async def correct_text(input: TextInput):
    result = grammar_agent.correct(input.text)
    return {"input": input.text, "correction": result}
