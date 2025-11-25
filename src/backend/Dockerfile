FROM python:3.10.18-slim

# System deps you wanted to keep
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# Install backend dependencies
COPY src/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2) Create src package folder inside container
RUN mkdir -p /app/src

# 3) Copy only what backend actually needs
COPY src/backend /app/src/backend
COPY src/models /app/src/models
COPY src/logger_config.py /app/src/logger_config.py
COPY src/__init__.py /app/src/__init__.py

# 4) Copy CLIP weights
COPY clip_weights /app/clip_weights

EXPOSE 8000

# 4) Run FastAPI using the same module path your code expects
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


