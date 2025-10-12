from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from ..utils.prompts import GRAMMAR_PROMPT

class GrammarAgent:
    def __init__(self):
        # Replace model_name with whichever provider/model you use
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        self.template = GRAMMAR_PROMPT

    def correct(self, text: str) -> str:
        prompt = self.template.format(text=text)
        # Using LangChain ChatOpenAI convenience method:
        response = self.llm.generate([{"content": prompt}])
        # response is a Generations object; extract string
        try:
            return response.generations[0][0].text
        except Exception:
            # Fallback for different langchain versions
            return str(response)
