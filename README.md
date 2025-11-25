git init
git remote add origin https://github.com/<your-username>/Advanced_prd_RAG.git
git pull origin main --allow-unrelated-histories
git add .
git commit -m "Initial project upload"
git branch -M main
git push -u origin main
sudo apt install -y tree
tree -L 4

streamlit run src\frontend\app.py
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 
C:\Users\User\AppData\Local\Temp\tmpx22pd3.pdf
Explanation:

python -m uvicorn → launch uvicorn through python

src.agentic_rag.backend.main:app → path to your FastAPI app

--host 0.0.0.0 → allow calls from Streamlit

--port 8000 → listen on port 8000

--reload → autoreload code when you edit

C:\Users\User\AppData\Local\Temp\tmpx22pd3.pdf

#pipreqs . --force --savepath=requirements.txt



'''''''''''''''''''''''''''''''''''''''''''''''''''''''

🚀 Advanced Agentic RAG System
FastAPI + Streamlit + NGINX + Docker + Google OAuth + SSL + AstraDB

A complete production-ready Retrieval-Augmented Generation (RAG) platform built with:

🧠 Agentic LLM Pipeline

🔍 Astra DB Vector Search

🎨 Multimodal: Text + PDF + Images

👤 Google OAuth Login

📦 FastAPI Backend

🌐 Streamlit Frontend

🛡️ NGINX Reverse Proxy

🔒 HTTPS (Let’s Encrypt + Certbot)

🐳 Docker Compose Deployment

☁️ Full EC2 Deployment Guide

This README provides end-to-end setup, including Docker, domain setup, SSL, and server configuration.

📚 Table of Contents

1. Project Overview
2. Architecture
3. Features
4. Folder Structure
5. Environment Variables

Local Setup (Docker)

Production Deployment on AWS EC2

Create EC2

Install packages

Clone project

Configure NGINX

SSL setup

Start Docker services

NGINX Configuration

SSL Certificate Setup

Testing the Deployment

Troubleshooting

Useful Commands

Future Enhancements

🧠 1. Project Overview

This project is a fully functional RAG system that supports text, PDFs, PowerPoints, Excel, images, and embeddings.
The system stores processed chunks in AstraDB Vector DB and uses a custom Agent LLM pipeline to answer questions.

Users authenticate using Google OAuth2, then interact with the Streamlit UI, upload documents, and ask questions.

🏗️ 2. Architecture
                       🌍 Internet Users
                               │
                               │  HTTPS (Port 443)
                               ▼
                   ┌──────────────────────────────┐
                   │     NGINX (Host on EC2)       │
                   │    /etc/nginx/sites-enabled/  │
                   │    SSL via Certbot            │
                   └──────────────┬───────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
 / (Frontend UI)        /api/* (Backend API)       WebSocket/TLS Upgrade
 Proxy to 8501          Proxy to 8000              (Streamlit Live App)
       │                          │
       ▼                          ▼
┌──────────────────┐      ┌────────────────────────┐
│ Streamlit Frontend│      │   FastAPI Backend      │
│ Docker Container  │      │ Docker Container       │
│ Port 8501         │      │ Port 8000              │
└──────────────────┘      └────────────────────────┘
       │                          │
       │  JWT Token (from Google) │
       │                          │
       ▼                          ▼
   Session State           User Collections,
                           Agents, Memory,
                         Vector Retriever, DB Index

                                 │
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   AstraDB Vector Store │
                    │  (Embeddings, Chunks)  │
                    └────────────────────────┘

                                 │
                                 ▼
                     ┌─────────────────────┐
                     │   LLM (Euri Model)  │
                     │  Agent + Tools      │
                     └─────────────────────┘


🚀 3. Features
🧠 Agentic Intelligence

Multi-vector retrieval

LLM-based query rewriting

Image-based summaries

Memory-aware responses

📄 Document Handling

Supports:

PDF (OCR + poppler + tesseract)

DOCX

PPTX

XLSX

CSV / TXT

🔍 Vector Search

AstraDB Vector Store

Record Manager + MultiVectorRetriever

🔐 Authentication

Google OAuth

JWT token

Secure cookie session

💻 Deployment-Ready

Docker Compose

NGINX reverse proxy

HTTPS using Certbot

Optimized for EC2

📁 4. Folder Structure
Advanced-RAG/
├── docker-compose.yml
├── nginx/
│   └── default.conf
├── src/
│   ├── backend/
│   │   ├── main.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── DB.py
│   │   ├── agent.py
│   │   ├── Adding_files.py
│   │   ├── chunking_retrieveing.py
│   │   ├── hybrid_pdf_parser.py
│   │   ├── image_processing_bytes.py
│   │   ├── utilis.py
│   ├── frontend/
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   ├── requirements.txt
│   ├── models/
│   └── logger_config.py
└── README.md

🔐 5. Environment Variables

Create:

src/backend/.env


Content:

GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxxx
JWT_SECRET=super-secret
ASTRA_DB_ID=xxxx
ASTRA_DB_REGION=xxxx
ASTRA_DB_APPLICATION_TOKEN=xxxx

🐳 6. Local Setup (Docker)
Build & Run:
docker-compose up -d --build

Access:

Frontend: http://localhost:8501

Backend: http://localhost:8000

☁️ 7. Production Deployment (AWS EC2)

PART 1 — Launch the EC2 INSTANCE

Launch EC2:
Ubuntu 22.04 LTS
t2.medium or t3.medium
Ports open: 22, 80, 443
Attach key pair

PART 2 — CONNECT TO EC2
ssh -i /path/to/key.pem ubuntu@EC2_PUBLIC_IP

PART 3 — INSTALL DOCKER & DOCKER COMPOSE
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker
docker --version
sudo apt install -y docker-compose
docker-compose --version

PART 4 — INSTALL HOST NGINX

sudo apt install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
sudo nginx -t

Test from browser:
http://EC2_IP -> you will ngnis page

PART 5 — PREPARE NGINX SSL DIRECTORIES
These ensure Certbot + NGINX have the correct directories.
sudo mkdir -p /etc/letsencrypt - This is where Certbot will store SSL certificates: fullchain.pem, privkey.pem
sudo mkdir -p /var/www/html - This is needed because Certbot uses HTTP-01 challenge, which requires: /var/www/html/.well-known/acme-challenge/<token>
sudo chown -R www-data:www-data /var/www/html - Certbot uses Nginx user www-data, so permissions must be correct.

🧩 PART 6 — UPLOAD PROJECT TO EC2

git clone <your_repo>
cd Advanced-RAG
Correct structure MUST be:

Advanced-RAG/
  docker-compose.yml
  nginx/default.conf  (Later not needed; we use host nginx)
  src/backend/Dockerfile
  src/frontend/Dockerfile
  src/backend/main.py
  src/frontend/app.py
  src/backend/requirements.txt
  src/frontend/requirements.txt


mkdir -p models/blobs
mkdir -p models/qwen2.5vl
mkdir -p clip_weights

PART 7 — DOCKER COMPOSE (NO NGINX IN DOCKER) Use this final docker-compose.yml:

PART 8 — UPDATE BACKEND DOCKERFILE - nano src/backend/Dockerfile

PART 9 — BUILD AND START DOCKER
docker-compose build backend
docker-compose up -d --build
docker ps


PART 10 — HOST NGINX CONFIG (WITHOUT SSL)
sudo nano /etc/nginx/sites-available/genaipoconline
paste :-
server {
    listen 80;
    server_name genaipoconline.online www.genaipoconline.online;

    # Streamlit UI
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
    }

    # Streamlit WebSockets
    location /_stcore/ {
        proxy_pass http://127.0.0.1:8501/_stcore/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
    }

    # FastAPI backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Authorization $http_authorization;
    }
}

Enable config:
sudo ln -s /etc/nginx/sites-available/genaipoconline /etc/nginx/sites-enabled/ - Enable NGINX site
sudo rm /etc/nginx/sites-enabled/default - remove deafult config
sudo nginx -t
sudo systemctl reload nginx

PART 11 — INSTALL CERTBOT SSL
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d genaipoconline.online -d www.genaipoconline.online

PART 12 — VERIFY EVERYTHING
Frontend:
https://genaipoconline.online

OAuth Login:
https://genaipoconline.online/api/login

Backend:
https://genaipoconline.online/api/health

Backend Health
curl https://genaipoconline.online/api/health



PART 13 — MANAGE BACKEND LOGS
docker logs -f backend
CTRL + C

🧩 PART 14 — REBUILD BACKEND (when code changes)
docker-compose build backend
docker-compose up -d

🧩 PART 15 — REBUILD FRONTEND (when code changes)

docker-compose build frontend
docker-compose up -d

🧩 PART 16 — RESTART EVERYTHING
docker-compose down
docker-compose up -d




🐳 DOCKER COMMANDS USED + MUST-KNOW DEBUG COMMANDS (Organized by category)

✅ 1. Container Status & Basic Monitoring
👉 Show running containers
docker ps

👉 Show ALL containers (running + stopped)
docker ps -a

👉 Show container logs
docker logs backend
docker logs frontend
docker logs nginx

👉 Follow logs in real time
docker logs -f backend

Exit using:

CTRL + C

✅ 2. Starting / Stopping / Restarting Services
👉 Start all containers (using docker-compose)
docker-compose up -d

👉 Stop all containers
docker-compose down

👉 Restart only backend
docker-compose restart backend

👉 Restart everything
docker-compose down
docker-compose up -d

✅ 3. Build / Rebuild Images
👉 Build ONLY backend image
docker-compose build backend

👉 Build ONLY frontend
docker-compose build frontend

👉 Build everything
docker-compose build

👉 Build + run everything
docker-compose up -d --build

✅ 4. Exec Into Running Container (very useful!)
👉 Get inside backend shell
docker exec -it backend bash

👉 Get inside frontend
docker exec -it frontend bash

👉 Get inside nginx
docker exec -it nginx bash


Inside the container, you can run:

pip list

ls

check installed commands like pdfinfo, tesseract

✅ 5. Troubleshooting Commands
👉 Check Docker service status
systemctl status docker

👉 Restart Docker engine
sudo systemctl restart docker

✅ 6. Inspect Commands
👉 Inspect container details
docker inspect backend

👉 Inspect networks
docker network ls
docker network inspect app-net

👉 Inspect images
docker images

✅ 7. Remove Stopped Containers / Images / Cache
👉 Remove ALL stopped containers
docker container prune

👉 Remove dangling images
docker image prune

👉 Remove everything unused (careful!)
docker system prune -a

✅ 8. Volume & Disk Debugging
👉 List volumes
docker volume ls

👉 Inspect a volume
docker volume inspect <volume_name>

👉 Remove unused volumes
docker volume prune

✅ 9. Kill Containers Manually

Sometimes a container freezes during logs:

👉 Find its PID
ps aux | grep docker

👉 Kill it
kill -9 <PID>

✅ 10. Port Conflicts Debugging

If you see:

ERROR: port 80 already in use


Check what is using port 80:

sudo lsof -i :80


Kill that process:

sudo kill -9 <PID>

