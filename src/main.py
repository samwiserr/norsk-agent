from src.agents.exam_agent import ExamAgent
from src.agents.grammar_agent import GrammarAgent
from dotenv import load_dotenv
load_dotenv()

def choose_mode() -> str:
    print("\nNorskAgent 🇳🇴  — Choose a mode:")
    print("  1) Evaluate (exam-style: corrected + explanation + tip)")
    print("  2) Grammar Fix (correct + short explanation)")
    print("  q) Quit")
    while True:
        choice = input("\nEnter 1, 2, or q: ").strip().lower()
        if choice in ("1", "2", "q"):
            return choice
        print("Please enter 1, 2, or q.")

def main():
    print("Welcome to NorskAgent 🇳🇴 (local model via Ollama)")
    exam_agent = ExamAgent(model="mistral")
    fix_agent = GrammarAgent(model="mistral")

    while True:
        mode = choose_mode()
        if mode == "q":
            print("Bye! 👋")
            break

        print("\nType a Norwegian sentence (or 'back' to choose another mode, 'quit' to exit):\n")
        while True:
            user_text = input("➡️  Your Norwegian sentence: ").strip()
            if user_text.lower() in ("quit", "exit"):
                print("Bye! 👋")
                return
            if user_text.lower() in ("back", "menu"):
                break
            if not user_text:
                continue

            try:
                if mode == "1":
                    feedback = exam_agent.evaluate(user_text)
                else:
                    feedback = fix_agent.fix(user_text)
                print("\n📝 Response:\n")
                print(feedback)
                print("\n" + "-" * 60 + "\n")
            except Exception as e:
                print("⚠️ Error:", e)
                print("Tip: Make sure Ollama is running and the model is pulled (e.g., `ollama pull mistral`).\n")

if __name__ == "__main__":
    main()
