# =============================
# 1. Base Image
# =============================
FROM python:3.10.18-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# =============================
# 2. System Dependencies
# =============================
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# =============================
# 3. Install Ollama
# =============================
RUN curl -fsSL https://ollama.com/install.sh | sh

# =============================

# =============================
# 5. Working Directory
# =============================
# → This ensures Uvicorn runs inside backend folder
WORKDIR /app/src/backend

# Add src root to Python module search path
ENV PYTHONPATH=/app/src


# 6. Copy project code (SRC) and requirements
# IMPORTANT: Build context must be project root
# =============================
# Copy the whole src folder so imports like `from src.models ...` work
COPY src /app/src

# Copy root-level requirements.txt
COPY requirements.txt /app/requirements.txt


# =============================
# 6. Python Packages
# =============================
# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# =============================
# 7. Expose FastAPI port
# =============================
EXPOSE 8000

# =============================
# 8. Start Ollama + FastAPI together
# =============================
CMD ollama serve & \
    sleep 5 && \
    ollama pull qwen2.5vl:3b && \
    uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
