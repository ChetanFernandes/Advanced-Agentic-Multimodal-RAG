FROM python:3.10.18-slim

WORKDIR /app

# Copy ONLY requirements first (to use Docker layer caching)
COPY src/frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 2) Create src package inside container
RUN mkdir -p /app/src

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