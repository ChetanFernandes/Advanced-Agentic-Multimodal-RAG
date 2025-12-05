git init
git remote add origin https://github.com/<your-username>/Advanced_prd_RAG.git
git pull origin main --allow-unrelated-histories
git add .
git commit -m "Initial project upload"
git branch -M main
git push -u origin main
sudo apt install -y tree
tree -L 4
-----------------------------------------------------
streamlit run src\frontend\app.py

python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 

python -m uvicorn → launch uvicorn through python

src.agentic_rag.backend.main:app → path to your FastAPI app

--host 0.0.0.0 → allow calls from Streamlit

--port 8000 → listen on port 8000

--reload → autoreload code when you edit

-----------------------------------------------------

pipreqs . --force --savepath=requirements.txt

----------------------------------------------------




