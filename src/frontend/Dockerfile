FROM python:3.10.18-slim

WORKDIR /app

# 2) Install dependencies first (best caching)
COPY src/frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) Copy only what frontend actually needs
COPY src/frontend /app/src/frontend
COPY src/logger_config.py /app/src/logger_config.py
COPY src/__init__.py /app/src/__init__.py
# If later frontend needs models too, we can add:
# COPY src/models /app/src/models

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]