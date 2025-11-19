cd /c/path/to/Advanced_prd_RAG
git init
git remote add origin https://github.com/<your-username>/Advanced_prd_RAG.git
git pull origin main --allow-unrelated-histories
git add .
git commit -m "Initial project upload"
git branch -M main
git push -u origin main


DEBUG
INFO
WARNING
ERROR
CRITICAL




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

langchain 0.3.27 depends on langchain-core<1.0.0 and >=0.3.72
langchain-text-splitters 0.3.11 depends on langchain-core<2.0.0 and >=0.3.75
langchain-astradb 0.3.3 depends on langchain-core<0.3 and >=0.1.31

streamlit run "src\frontend\app.py"
D:\GEN AI\GEN_AI_MASTERS_Bappy\Langchain\Advanced_prod_RAG
Explanation:

python -m uvicorn → launch uvicorn through python

src.agentic_rag.backend.main:app → path to your FastAPI app

--host 0.0.0.0 → allow calls from Streamlit

--port 8000 → listen on port 8000

--reload → autoreload code when you edit

C:\Users\User\AppData\Local\Temp\tmpx22pd3.pdf




DEBUG
INFO
WARNING
ERROR
CRITICAL


astrapy==2.1.0
clip_anytorch==2.6.0
fastapi==0.121.0
langchain==1.0.3
langchain_astradb==1.0.0
langchain_community==0.4.1
langchain_core==1.0.3
langchain_unstructured==0.1.6
numpy==2.3.4
openpyxl==3.1.5
pandas==2.3.3
Pillow==12.0.0
python-dotenv==1.2.1
python_docx==1.2.0
python_pptx==0.6.23
Requests==2.32.5
streamlit==1.49.1
torch==2.8.0+cu126
unstructured==0.10.30
utility==1.0



ECR URI - 200937443183.dkr.ecr.eu-north-1.amazonaws.com/rag 
13.51.107.191 - public IP