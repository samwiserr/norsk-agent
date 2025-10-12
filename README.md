# 🇳🇴 Norsk Agent – AI-Powered Norwegian Language Exam Assistant

**Norsk Agent** is an intelligent AI system designed to help learners prepare for the **Norwegian A1–B1 exams**.  
It uses *agentic AI* concepts — each specialized agent (Exam, Grammar, and Scorer) collaborates to correct, explain, and rate user answers in real time.

---

## 🧠 Overview

| Agent | Purpose |
|-------|----------|
| **GrammarAgent** | Corrects grammar and spelling, provides simple explanations in English. |
| **ExamAgent** | Acts like a Norwegian examiner — gives feedback, tips, and corrections. |
| **ScorerAgent** | Evaluates CEFR level (A1–B1) and gives a numeric score with rationale. |

All three agents run locally via **Ollama** using open models such as `llama3.2:1b` or `llama3.2:3b`.

---

## 🧰 Tech Stack

- **Python 3.11+**
- **FastAPI** (API layer)
- **LangChain + langchain-ollama**
- **Ollama** (local LLM inference)
- **SQLite** (interaction logs)
- **Streamlit** (frontend dashboard)
- **Docker** + `docker-compose`

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- Install [Ollama](https://ollama.ai) and pull a model:
  ```bash
  ollama pull llama3.2:3b
