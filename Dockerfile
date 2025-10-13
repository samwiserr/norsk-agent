FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn streamlit requests pandas langchain langchain-community

# Copy project files
COPY . /app

# Expose API and UI ports
EXPOSE 8000 8501

# Allow container to talk to host's Ollama
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Start FastAPI and Streamlit using JSON CMD
CMD ["bash", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501"]
