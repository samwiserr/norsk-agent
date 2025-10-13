# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# (Optional) install curl for debugging health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Normalize potential Windows line endings on start.sh, then make it executable
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Expose API and Streamlit
EXPOSE 8000 8501

# Talk to host's Ollama from inside the container
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Start both services
CMD ["/app/start.sh"]
