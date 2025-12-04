# FastAPI is built on top of asyncio, a Python library that supports asynchronous programming — 
# meaning tasks can run without blocking each other.

from fastapi import FastAPI, Query, UploadFile, File, Form, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager
from src.backend.DB import ConnectToAstraDB
from src.backend.agent import web_agent
from src.backend.Adding_files import Adding_files_DB
from src.backend.utilis import *
from src.backend.image_processing_bytes import extract_Image_summaries
from src.models import *
import asyncio
from langchain_ollama import ChatOllama
from src.backend.chunking_retrieveing import question_answering
from fastapi.responses import JSONResponse
from src.logger_config import log
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
import jwt
from datetime import datetime, timedelta, timezone
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
import shutil

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretjwt")
JWT_ALGO = "HS256"

def create_jwt(user_info: dict):  # creates a JWT (signed token) containing user info
    payload = {
        "user": user_info,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def verify_jwt(request: Request):
    auth = request.headers.get("authorization")
    log.info(f"auth header -> {auth}")
    if auth is None or not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")
    
    token = auth.split(" ")[1]

    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return decoded["user"]  # return user info
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except Exception:
        raise HTTPException(401, "Invalid token")

# CREATE FASTAPI APP (ONCE)

@asynccontextmanager # Tells - “Hey, when the app starts, run this function once before handling any user requests.”
async def lifespan(app: FastAPI):
    try:

        app.state.user_collections = {}
        log.info("Server is starting up...")

        app.state.ASTRA_DB = ConnectToAstraDB()

        #astra_index = await asyncio.to_thread(app.state.ASTRA_DB.add_index)
        app.state.llm = ChatOllama(model="qwen2.5vl:3b", base_url=os.getenv("OLLAMA_HOST"), temperature = 0.2)

        #app.state.llm = ChatOllama(model="qwen2.5vl:3b", temperature = 0.5)
        #app.state.llm = EuriLLM()
        app.state.web_search_agent = web_agent(app.state.llm)


        log.info("Startup initialization complete")

        # Yield control back to FastAPI (server runs after this)
        yield #→ tells FastAPI “do the startup code before yield, then run the server; after the server stops, run shutdown code after yield”

        # SHUTDOWN (runs once when app stops)
        log.info("Shutting down gracefully...")

        # Optionally: close DB connections or cleanup models

    except Exception:
         log.exception("App Intilization failed")

app = FastAPI(lifespan=lifespan) # router is the internal object FastAPI uses to manage all endpoints (routes).

@app.get("/")
async def root():
    return {"message": "Backend is running"}

@app.get("/health")
async def health(request : Request):
    return JSONResponse({'success': 'Backend is healthy'},status_code=200)
     

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "supersecret")
)

origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://genaipoconline.online",
    "http://genaipoconline.online",
]


# Cross-Origin Resource Sharing.
app.add_middleware(CORSMiddleware,
                   allow_origins = origins, # “Any website is allowed to talk to my API.”
                   allow_credentials = True, 
                   allow_methods = ["*"], # “Streamlit can call GET, POST, DELETE, PUT — any HTTP method.”
                   allow_headers = ["*"],# “Requests are allowed to send custom headers.”
                   )


oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

@app.get("/login")
async def login(request: Request):
    """Redirect user to Google OAuth2 consent screen"""
    try:
        log.info("Inside login function")
        redirect_uri = "https://genaipoconline.online/api/auth/callback"
        #redirect_uri = "http://localhost:8000/auth/callback"
        log.info(f"redirect_uri - {redirect_uri}")
        return await oauth.google.authorize_redirect(request, redirect_uri)
    except Exception:
        log.exception("Login failed")
    

@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Google redirects here after login"""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")

        if not user_info:
            return JSONResponse({"error": "Login failed"}, status_code=400)
        
        # Create JWT
        jwt_token = create_jwt(dict(user_info))
   

        # Auto-submit POST form to Streamlit
        #form method="GET" action="https://genaipoconline.online/">
        #form method="GET" action="http://127.0.0.1:8501">
        html = f"""
        <html>
        <body onload="document.forms[0].submit()">
        <form method="GET" action=https://genaipoconline.online/>
                <input type="hidden" name="token" value="{jwt_token}">
            </form>
            Redirecting...
        </body>
        </html>
        """
        return HTMLResponse(html)
    except Exception:
        log.exception("Login failed")


@app.get("/available_sources")
async def available_sources(user: dict = Depends(verify_jwt)):
        '''
        if not user:
            return RedirectResponse(url="/login")
        '''
        user_id = user["sub"].strip() # Google unique ID
        log.info(f"Authenticated user: {user['email']} ({user_id})")
        try:
            if user_id not in app.state.user_collections:
                app.state.user_collections[user_id] = {}
            
            vector_store = app.state.user_collections[user_id].get("vector_store", [])
            if not vector_store:
                return JSONResponse(content={"sources": []},status_code=404)
            if vector_store:
                vector_store = app.state.user_collections[user_id].get("vector_store")
                all_docs = await vector_store.asimilarity_search("", k=1000)
                app.state.available_doc_names = list({doc.metadata.get("source") for doc in all_docs if doc.metadata.get("source")})
                log.info(f"available_doc_names -> {app.state.available_doc_names}")
                return JSONResponse(content={"sources": app.state.available_doc_names,"user_d" :user_id},status_code = 200)
        except Exception:
            log.exception("Failed to get available resources")
            return JSONResponse(
                content={"sources": [], "message": "Failed to get available resources"}, status_code=500)


MAX_UPLOAD_MB = 200
@app.post("/upload_file")
async def upload_file(user: dict = Depends(verify_jwt), file: UploadFile = File(...)):
    try:
        log.info(f"Decoded user {user}")
        if not user:
            return RedirectResponse(url="/login")
        user_id = user["sub"]  # Google unique ID
        log.info(f"user_id - {user_id}")
        collection_name = f"{user_id}_collection"
        
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB *1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB")

        astra_index = await asyncio.to_thread(app.state.ASTRA_DB.add_index, collection_name)
        vector_store = astra_index["vector_store"]
        vector_retriever = astra_index["vector_retriever"]
        record_manager = astra_index["record_manager"]
        collection_name = astra_index["collection_name"]
        log.info(f"vector_store -{vector_store}")

        if not hasattr(app.state, "user_collections"):
            app.state.user_collections = {}

        if user_id not in app.state.user_collections:
            app.state.user_collections[user_id] = {}
            log.info(f"Created new entry for user: {user_id}")

        user_data = app.state.user_collections[user_id]

        user_data.update(astra_index)

        file.file.seek(0)
        log.info(f"Uploading file: {file.filename}, size={len(data)}")
        log.info(f"user_id -> {user_id}")

        file_name = file.filename
        file_bytes = data

        print(f"Received upload from user_id: {user_id}")
        print(f"Uploaded file: {file.filename}")

        
        DB_process = Adding_files_DB(vector_retriever,record_manager,vector_store,file_name,file_bytes,user_id)
        
        result = await asyncio.to_thread(DB_process.in_memory_store)

        if not result or result.get("status") == "error":
             msg = result.get("message", "Failed to process file.")
             log.exception(f"Failed to process file {msg}")
             return JSONResponse(content={"message": msg}, status_code=400)
        
        return JSONResponse(
            content={"message": result.get("message")},status_code=200
        )
    except Exception:
        log.exception("File upload failed")
        return JSONResponse(content={"message": "Internal error during upload."}, status_code=500)



        
@app.post("/query")
async def query_endpoint(query:str = Form(...), selected_doc:str = Form(None), image:  UploadFile = File(None), user=Depends(verify_jwt)):
    if not user:
        return RedirectResponse("/login")
    user_id = user['sub']
    try:
        log.info(f"query asked by user - {query}")
        log.info(f"Checking if user is registered : {user_id}")
        if user_id not in app.state.user_collections:
            app.state.user_collections[user_id] = {}
            user_data = app.state.user_collections[user_id]
            log.info(f"User is not registered")
            log.info(f"Created new entry for user")

        else:
            log.info(f"User is already registered : {user_id}")
            user_data = app.state.user_collections[user_id]

        # Initialize agent + memory if missing

        log.info(f"Checking if user is assigned with agent and memeory")
        if "agent" not in user_data or "memory" not in user_data:
            log.info(f"Agent and memory not assigned. Assigning new agent & memory to user")
            agent, memory = app.state.web_search_agent.initializing_agent()
            user_data["agent"] = agent
            user_data["memory"] = memory
        else:
            log.info(f"User has already assigned agent and memory")
            agent = user_data["agent"]
            memory = user_data["memory"]

        
        image_summary = ""                          
        if not selected_doc:
            log.info("Document not selected. Pass query directly to agent on hitting submit button")
            if image:
                log.info("Documument not selected but image selected. Extract image summary and pass to agent query and image summarry")
                image.seek(0)  
                image_bytes = await image.read()   # Reads the entire file as bytes (async)
                content_type = image.content_type
                log.info(type(image_bytes))           # <class 'bytes'>
                log.info(f"Image file_name - {image.filename}")            
                log.info(f"Image content type - {image.content_type}") # "image/png"
                image_summary = await extract_Image_summaries(image_bytes,content_type)
                log.info(f"Image suammry extracted - {image_summary}")
                if image_summary:
                     image_summary = "\n".join(image_summary) if isinstance(image_summary, list) else str(image_summary)
                final_response, memory = await app.state.web_search_agent.query_answering_async(agent,query,image_summary,memory)
        
            else:
                log.info("Passing query to agent for final results")
                final_response, memory = await app.state.web_search_agent.query_answering_async(agent,query,image_summary,memory)

        else:
            log.info(f"pre query {app.state.user_collections}")
            vector_store = app.state.user_collections[user_id].get("vector_store")
            vector_retriever = app.state.user_collections[user_id].get("vector_retriever")
            log.info(f"VS -> {vector_store}")

            log.info("Initilazing question ansering class")
            QA = question_answering(app.state.llm,vector_store,vector_retriever,selected_doc)

            log.info("Entering Question answering class to extract results for query")
            results,text_query  = await QA.extract_question_from_given_input(query,image)

            log.info("Passing retrived results and query to agent for final results")
            final_response, memory = await app.state.web_search_agent.query_answering_async(agent,text_query,results,memory)

   
        log.info(f"Final Answer\n, {final_response}")
        log.info(f"\n[DEBUG] Memory for user {user_id} - {memory}:")
        for idx, msg in enumerate(memory.chat_memory.messages, 1):
            sender = "USER" if msg.type == "human" else "ASSISTANT"
            log.info(f"{idx}. [{sender}]: {msg.content}")

        return JSONResponse(content={"result": f"{final_response}"}, status_code=200)    

    except Exception:
            log.exception("Failed to answer question")
            return JSONResponse(content={"status": f"Failed to answer question (Internal_error)"},status_code=500)
    


#### Deletion and logout
    
def delete_user_folder(user_id, base_dir="all_images"):
    """
    Deletes the entire user folder (e.g., all_images/<user_id>) safely.
    Example:
        delete_user_folder("123")
    """
    try:
        user_dir = os.path.join(base_dir, str(user_id))
        
        if not os.path.exists(user_dir):
            log.info(f"⚠️ No such user folder: {user_dir}")
            return

        # Safety guard — ensure we're really deleting inside all_images
        if os.path.basename(os.path.dirname(user_dir)) != os.path.basename(base_dir):
            log.info(f"⚠️ Unsafe path detected: {user_dir}")
            return

        shutil.rmtree(user_dir)
        log.info(f"Deleted user folder: {user_dir}")

    except Exception as e:
        log.exception(f"❌ Failed to delete user folder {user_id}")


@app.delete("/delete_collection")
async def delete_user_data(user_id: str = Query(...)):
    try:
        log.info("Inside deletion function")
        log.info(f"Pre-deletion: {app.state.user_collections}")
        if user_id not in app.state.user_collections:
            return JSONResponse(
                content={"message": "No collection exists for user"},
                status_code=404
            )
        
        user_data = app.state.user_collections[user_id]
         # Delete vector store if exists
        vector_store = user_data.get("vector_store")
        collection_name = user_data.get("collection_name")
        
        if vector_store:
            await vector_store.adelete_collection()


        user_data.pop("vector_store", None)
        user_data.pop("collection_name", None)
        user_data.pop("record_manager", None)
        user_data.pop("vector_retriever", None)

        delete_user_folder(user_id)
        log.info(f"Post deletion: {app.state.user_collections}")
        return JSONResponse(
            content={"message": f"Collection {collection_name} deleted successfully"},
            status_code=200,
        )
    except Exception:
        log.exception("Collection deletion failed")
        return JSONResponse(
            content={"message": "Failed to delete collection"},
            status_code=500
        )
  
         
@app.post("/logout")
async def logout(user: dict = Depends(verify_jwt)):
    """Logout and clear session"""
    try:
        log.info('Entering log out session')
        user_id = user['sub']
        if user_id in app.state.user_collections:
            log.info(f"Pre-logout: {app.state.user_collections[user_id]}")
        # Clean only agent+memory (not vector DB)
            if "agent" in app.state.user_collections[user_id]:
                del app.state.user_collections[user_id]["agent"]
            if "memory" in app.state.user_collections[user_id]:
                del app.state.user_collections[user_id]["memory"]
            log.info(f"Post-logout: {app.state.user_collections[user_id]}")
            return JSONResponse(
                                content={"message": "Successfully logged out"},
                                status_code=200)
        else:
            raise HTTPException(401, "Please login again.")
    except Exception:
        log.exception("Log out failed")
        raise HTTPException(401, "Please login again.")













  
''' 
log.info("\n[DEBUG] Memory so far:")
for idx, msg in enumerate(app.state.memory.chat_memory.messages, 1):
    sender = "USER" if msg.type == "human" else "ASSISTANT"
    log.info(f"{idx}. [{sender}]: {msg.content}")
'''

'''
1️⃣ What is app: FastAPI in lifespan(app: FastAPI)?

When FastAPI calls your lifespan function, it passes the actual app instance.

app is the same object used for routing, middleware, and state throughout your server.

This means anything you attach to app will be accessible anywhere in your FastAPI app (endpoints, middleware, background tasks).

2️⃣ What is app.state?

app.state is like a storage locker attached to the FastAPI app object.

You can store any Python object here: LLMs, agents, DB connections, caches, etc.

Anything in app.state is shared across all requests, unlike local variables inside endpoints.


App instance (FastAPI)
│
├── router (manages all endpoints / routes)
│     └── lifespan_context → runs startup/shutdown
├── middleware
└── state
All endpoints go through router
router.lifespan_context ensures startup code runs before any endpoint
So every request actually goes through app.router.
'''

    