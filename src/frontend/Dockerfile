FROM python:3.10.18-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

''' 
ENV Setting	Why It is Needed
HEADLESS=true	So Streamlit doesn't try to open a local browser — required in Docker/servers.
PORT=8501	Defines the internal container port Streamlit will run on.
GATHER_USAGE_STATS=false	Disable telemetry / unnecessary network requests.
'''