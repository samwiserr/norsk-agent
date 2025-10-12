FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn streamlit requests pandas langchain langchain-community

# Copy project
COPY . /app

# Expose API and UI ports
EXPOSE 8000 8501

# IMPORTANT: let the container talk to host's Ollama
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Start API and Streamlit
CMD bash -lc "uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
              streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501"