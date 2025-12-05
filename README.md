🌟 Project Title:- 

   Advanced Agentic RAG System + ChatGPT-Powered Real-Time Chatbot

  A complete production-ready Retrieval-Augmented Generation (RAG) and Chat_GPT Model platform built with:

      🧠 Agentic LLM Pipeline
      🔍 Astra DB Vector Search
      🎨 Multimodal: Text + PDF + Images
      👤 Google OAuth Login
      📦 FastAPI Backend
      🌐 Streamlit Frontend
      🛡️ NGINX Reverse Proxy
      🔒 HTTPS (Let’s Encrypt + Certbot)
      🐳 Docker Compose Deployment
      🌐 Ngrok Support
      ☁️ AWS EC2 

This README provides an end-to-end guide covering architecture, features, setup, deployment, SSL automation, troubleshooting, and DevOps workflow.

📚 Table of Contents:-


      1. Project Overview
      2. Architecture
      3. Features of RAG
      4. Folder Structure
      5. Environment Variables
      6. Local Setup (Docker)
      7. Production Deployment (AWS EC2)
      8. NGINX Configuration
      9. SSL Certificate Setup
      10. Testing the Deployment
      11. Useful Docker Commands
      12. Troubleshooting
    
🧠 1. Project Overview:-

      - This is a fully production-ready **RAG + Real-Time ChatGPT Chatbot System**.
      - It supports PDFs, PowerPoints, Excel files, images, and text documents, converts them into embeddings, and stores them in **AstraDB Vector DB** 
      - and uses a custom Agent LLM pipeline to answer questions. 
      - Users log in via **Google OAuth2**, upload documents through the **Streamlit UI**, and interact with a **ChatGPT-like conversational interface**
    Core components:
      a. AstraDB Vector Store for embeddings + retrieval
      b. Google OAuth2 for secure login
      c. FastAPI backend for processing documents, chunking, embeddings, and agent execution
      d. Streamlit frontend for user interaction
      e. NGINX reverse proxy with full HTTPS
      f. Docker Compose for local and production builds
      g. AWS EC2 deployment guide included

🏗️ 2. Architecture:-

<img width="381" height="655" alt="image" src="https://github.com/user-attachments/assets/6cba5fcb-a1eb-488a-8b94-c0e3ef5033fc" />

FULL REQUEST–RESPONSE LIFECYCLE

1. User enters message → hits NGINX
2. NGINX forwards to Frontend / Backend
3. Backend authenticates & processes query
4. Backend retrieves relevant chunks from AstraDB
5. Agentic pipeline prepares an optimized LLM prompt
6. Backend sends LLM prompt → Ngrok → Qwen
7. Qwen generates LLM response tokens
8. Response returns → Ngrok → Backend
9. Backend streams final answer to Streamlit
10. User sees real-time chat response
    
🚀 3. Features of RAG
 
      🧠 Agentic Intelligence
        a. Multi-vector retrieval
        b. LLM-based query rewriting
        c. Reranking retrived results
        d. Memory-aware conversation
        e. Image + PDF + text interpretation

      📄 Document Handling
      Supports:
        a. PDF (OCR via Tesseract + Poppler)
        b. DOCX
        c. PPTX
        d. XLSX
        e. TXT / CSV
        f. Images (CLIP embeddings)

      🔍 Vector Search
        a. AstraDB vector index
        b. Record Manager + MultiVectorRetriever
        c. Chunk metadata tracking
      
      🔐 Authentication
        a. Google OAuth
        b. JWT token
        c. Secure session cookies

      💻 Deployment Ready
        a. Docker Compose
        b. Host-level NGINX
        c. HTTPS via Certbot
        d. EC2 optimized

      🌐 Ngrok Support (Public URL for Local Testing)
        a. Expose local backend or Streamlit via secure tunnels
        b. Enables OAuth callback testing
        c. Great for development demo environments

📁 4. Folder Structure

<img width="298" height="542" alt="image" src="https://github.com/user-attachments/assets/17131332-88fc-4a32-b7eb-c14b4bc8d126" />

🔐 5. Environment Variables


        a. SERPER_API_KEY = ********************
        b. Google_API_KEY = ********************
        c. ASTRA_DB_APPLICATION_TOKEN = ********************
        d. ASTRA_DB_API_ENDPOINT = ********************
        e. LANGSMITH_TRACING = ********************
        f. LANGSMITH_ENDPOINT = ********************
        g. LANGSMITH_API_KEY = ********************
        h. LANGSMITH_PROJECT =  ********************
        i. GOOGLE_CLIENT_ID = ********************
        j. GOOGLE_CLIENT_SECRET = ********************
        k. SECRET_KEY = ********************
        l. JWT_SECRET = ********************
        m. OLLAMA_HOST = ********************
        n. OLLAMA_LOCAL_HOST = ********************

📸 Screenshots of Application


🐳 6. Local Setup (Docker)

     Build & Run:
        a. docker-compose up -d --build

☁️ 7. Production Deployment (AWS EC2)

    PART 1 — Launch the EC2 INSTANCE
           1. Ubuntu 22.04 LTS

    PART 2 — CONNECT TO EC2
           1. chmod 400 "AG.pem"
           2. ssh -i RAG.pem ubuntu@public_ip

    PART 3 — INSTALL DOCKER & DOCKER COMPOSE
      a. Step 1: Add Docker’s official GPG key
          1. sudo apt update
          2. sudo apt install -y ca-certificates curl gnupg
          3. sudo install -m 0755 -d /etc/apt/keyrings
          4. curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
          5. sudo chmod a+r /etc/apt/keyrings/docker.gpg

      b. Step 2: Add Docker repo
          1. echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
            https://download.docker.com/linux/ubuntu \
            $(lsb_release -cs) stable" | \
            sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

      c. Step 3: Update apt
          1. sudo apt update

      d. Step 4: Install Docker correctly
          1. sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
          2. docker --version
          3. docker compose --version
          4. sudo systemctl enable docker
          5. sudo systemctl start docker
          6. sudo systemctl status docker
          7. sudo usermod -aG docker ubuntu
          8. newgrp docker
      
    PART 4 — INSTALL HOST NGINX
          1. sudo apt install -y nginx
          2. sudo systemctl enable nginx
          3. sudo systemctl start nginx
          4. sudo nginx -t
          5. Test from browser:
              http://EC2_IP -> you will ngnis page
          6. sudo tail -f /var/log/nginx/access.log
      
    PART 5 — PREPARE NGINX SSL DIRECTORIES - These ensure Certbot + NGINX have the correct directories.
          1. sudo mkdir -p /etc/letsencrypt - This is where Certbot will store SSL certificates: fullchain.pem, privkey.pem
          2. sudo mkdir -p /var/www/html - This is needed because Certbot uses HTTP-01 challenge, which requires: /var/www/html/.well-known/acme-challenge/<token>
          3. sudo chown -R www-data:www-data /var/www/html - Certbot uses Nginx user www-data, so permissions must be correct.

    PART 6 — UPLOAD PROJECT TO EC2 + Models (clip)
          1. git clone <your_repo>
          2. cd Advanced-RAG -> Move to project root directory
          3. scp -i RAG.pem -r \
            "/d/GEN AI/GEN_AI_MASTERS_Bappy/Langchain/Advanced_prod_RAG/clip_weights" \
            ubuntu@51.21.169.168:/home/ubuntu/Advanced-RAG/
        
    PART 7 — BUILD AND START DOCKER
          1. docker compose up -d --build
          2. docker ps


🌐 8. NGINX Configuration


          1. sudo nano /etc/nginx/sites-available/genaipoconline.conf
              Paste:- 
              server {
                        listen 80;
                        server_name genaipoconline.online www.genaipoconline.online;

                        root /var/www/html;
                        index index.html index.htm;
                      }

          2. Enable config:
              a. sudo ln -s /etc/nginx/sites-available/genaipoconline.conf /etc/nginx/sites-enabled/ - Enable NGINX site
              b. sudo rm /etc/nginx/sites-enabled/default - remove deafult config
              c. sudo nginx -t
              d. sudo systemctl reload nginx

🔒 9. SSL Certificate Setup - Enable HTTPS using Certbot:


          1. sudo apt install -y certbot python3-certbot-nginx
          2. sudo certbot --nginx -d genaipoconline.online -d www.genaipoconline.online
          3. Final nginx config file can be checked in project root  - nginx\default.conf

🧪 10. Testing the Deployment


          1. Frontend:  https://genaipoconline.online
          2. OAuth Login: https://genaipoconline.online/api/login
          3. Backend Health: https://genaipoconline.online/api/health
          4. Local Tests:
             1. curl http://127.0.0.1:8501
             2. curl http://127.0.0.1:8000/health

🐳 11. Useful Docker Commands


      a. Monitoring
          1. docker ps
          2. docker ps -a  -  Show ALL containers (running + stopped)
          3. docker logs backend
          4. docker logs -f backend - Follow logs in real time. Exit using: CTRL + C
          5. docker logs frontend 
          6. docker logs -f frontend - Follow logs in real time
          7. sudo tail -f /var/log/nginx/access.log - Nginx logs
          
      b. Build & Restart
          1. docker compose build - Build everything
          2. docker compose build backend
          3. docker compose up -d --build
          4. docker compose restart backend
          5. docker compose build frontend
          6. docker compose up -d --build
        
      c. RESTART EVERYTHING
          1. docker compose down - Stop all containers
          2. docker compose up -d
          3. docker compose restart backend

      d. Shell into Containers
          1. docker exec -it backend bash
          2. docker exec -it frontend bash
          3. docker exec -it nginx bash
          4. Inside the container, you can run: pip list ls
      
      e. Troubleshooting Commands
          1. sudo systemctl restart docker
          2. docker inspect backend - Inspect container
          3. docker network ls -  Inspect networks
          4. docker network inspect app-net
          5. docker images - Inspect images
      
      f. Cleanup
          1. docker container prune -  Remove ALL stopped containers
          2. docker image prune - Remove dangling images
          3. docker system prune -a - Remove everything unused (careful!)

🛠️ 12. Troubleshooting


          1. docker volume ls - List all volumes.
          2. docker volume inspect - Inspect volume details.
          3. docker volume prune - Remove unused volumes.
          4. ps aux | grep docker - List stuck Docker processes.
          5. sudo kill -9 <PID> - Kill stuck Docker processes.
          6.sudo lsof -i :80 - List ports used by Docker services.

Tracking using Langsmith:-


| Metric                    | Value            | Meaning                          |
| ------------------------- | ---------------- | -------------------------------- |
| **Average Response Time** | **3.25 seconds** | Most responses delivered fast    |
| **Fastest (P50 FTT)**     | **1.05 sec**     | First token appears quickly      |
| **Slowest (P99)**         | **13–14 sec**    | Only 1% of requests are slow     |

🎉 Author - 

Chetan Fernandis - Full-Stack GenAI Engineer • RAG Systems • LLMOps • MLOps













