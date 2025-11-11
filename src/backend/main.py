# FastAPI is built on top of asyncio, a Python library that supports asynchronous programming — 
# meaning tasks can run without blocking each other.

from fastapi import FastAPI, Query, UploadFile, File, Form 
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager
from backend.DB import ConnectToAstraDB
from backend.agent import web_agent
from backend.Adding_files import Adding_files_DB
from backend.utilis import *
from backend.image_processing_bytes import extract_Image_summaries
from models import EuriLLM
import asyncio
from langchain_ollama import ChatOllama
from backend.chunking_retrieveing import question_answering
from fastapi.responses import JSONResponse
from src.logger_config import log
from astrapy import DataAPIClient

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins = ["*"],
                   allow_credentials = True,
                   allow_methods = ["*"],
                   allow_headers = ["*"],
                   )


@asynccontextmanager # Tells - “Hey, when the app starts, run this function once before handling any user requests.”
async def lifespan(app: FastAPI):
    try:

        app.state.user_collections = {}
        log.info("Server is starting up...")

        app.state.ASTRA_DB = ConnectToAstraDB()

        #astra_index = await asyncio.to_thread(app.state.ASTRA_DB.add_index)

        #app.state.llm = ChatOllama(model="qwen2.5vl:3b")
        app.state.llm = EuriLLM()
        app.state.web_search_agent = web_agent(app.state.llm)
        app.state.agent = await asyncio.to_thread(app.state.web_search_agent.initializing_agent)
        app.state.memory =  app.state.web_search_agent.memory



        log.info("Startup initialization complete")

        # Yield control back to FastAPI (server runs after this)
        yield #→ tells FastAPI “do the startup code before yield, then run the server; after the server stops, run shutdown code after yield”

        # SHUTDOWN (runs once when app stops)
        log.info("Shutting down gracefully...")

        # Optionally: close DB connections or cleanup models

    except Exception:
         log.exception("App Intilization failed")

    


app = FastAPI(lifespan=lifespan)
# router is the internal object FastAPI uses to manage all endpoints (routes).

@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), user_id: str = Query(...)):
    try:
        log.info(f"user_id - {user_id}")
        collection_name = f"{user_id}_collection"
        astra_index = await asyncio.to_thread(app.state.ASTRA_DB.add_index, collection_name)
        vector_store = astra_index["vector_store"]
        vector_retriever = astra_index["vector_retriever"]
        record_manager = astra_index["record_manager"]
        collection_name = astra_index["collection_name"]
        log.info(f"vector_store -{vector_store}")

        if not hasattr(app.state, "user_collections"):
            app.state.user_collections = {}

        app.state.user_collections[user_id] = astra_index
        log.info(f"{app.state.user_collections[user_id]}")


        data = await file.read()
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


@app.get("/available_sources")
async def available_sources(user_id : str = Query(...)):
        try:
            log.info("Entering available sources function")
            if not hasattr(app.state, "user_collections") or user_id not in app.state.user_collections:
                log.warning(f"No collection found for user_id={user_id}")
                return JSONResponse(
                    content={"sources": [], "message": f"No collection found for user {user_id}"},
                    status_code=404
            )

            vector_store = app.state.user_collections[user_id].get("vector_store")

            if not vector_store:
                log.warning(f"No vector_store found for user_id={user_id}")
                return JSONResponse(
                        content={"sources": [], "message": "Vector store not initialized"},
                        status_code=404
            )

            all_docs = await vector_store.asimilarity_search("", k=1000)
            app.state.available_doc_names = list({doc.metadata.get("source") for doc in all_docs if doc.metadata.get("source")})
            log.info(f"available_doc_names -> {app.state.available_doc_names}")
            return JSONResponse(content={"sources": app.state.available_doc_names},status_code = 200)
        
        except Exception:
            log.exception("Failed to get available resources")
            return JSONResponse(
                content={"sources": [], "message": "Failed to get available resources"},
                status_code=500
        )
          
 

@app.post("/query")
async def query_endpoint(query:str = Form(...), selected_doc:str = Form(None), image:  UploadFile = File(None), user_id: str = Query(...)):
    try:
        log.info(f"user_id - {user_id}")
        image_summary = ""
        if not selected_doc:
            log.info("Document not selected. Pass query directly to agent on hitting submit button")
            if image:
                image.seek(0)  
                image_bytes = await image.read()   # Reads the entire file as bytes (async)
                content_type = image.content_type
                log.info(type(image_bytes))           # <class 'bytes'>
                log.info(f"Image file_name - {image.filename}")            
                log.info(f"Image content type - {image.content_type}") # "image/png"
                image_summary = await extract_Image_summaries(image_bytes,content_type)
                if image_summary:
                     image_summary = "\n".join(image_summary) if isinstance(image_summary, list) else str(image_summary)
        
            log.info("Passing text query and image summary to agent for final results")
            final_response = await app.state.web_search_agent.query_answering_async(app.state.agent,query,image_summary)
            final_response = final_response.get("output", final_response)
            log.info(f"Final Answer\n, {final_response}")
            return JSONResponse(content={"result": f"{final_response}"}, status_code=200)     
        
        else:
            vector_store = app.state.user_collections[user_id].get("vector_store")
            vector_retriever = app.state.user_collections[user_id].get("vector_retriever")

            log.info("Initilazing question ansering class")
            answer = question_answering(app.state.llm,vector_store,vector_retriever,selected_doc)

            log.info("Entering Question answering class to extract results for query")
            results,text_query  = await answer.extract_question_from_given_input(query,image)

            log.info("Passing retrived results and query to agent for final results")
            final_response = await app.state.web_search_agent.query_answering_async(app.state.agent,text_query,results)
            final_response = final_response.get("output", final_response)
            log.info(f"Final Answer\n, {final_response}")

            log.info("\n[DEBUG] Memory so far:")
            for idx, msg in enumerate(app.state.memory.chat_memory.messages, 1):
                sender = "USER" if msg.type == "human" else "ASSISTANT"
                log.info(f"{idx}. [{sender}]: {msg.content}")
            return JSONResponse(content={"result": f"{final_response}"}, status_code=200)    

    except Exception:
            log.exception("Failed to answer question")
            return JSONResponse(content={"status": f"Failed to answer question (Internal_error)"},status_code=500)
    


     
@app.delete("/delete_collection")
async def delete_user_data(user_id: str = Query(...)):
    try:
         log.info("Inside deletion function")
         log.info(f"pre Deletion {app.state.user_collections}")
         vector_store = app.state.user_collections[user_id].get("vector_store")
         collection_name = app.state.user_collections[user_id].get("collection_name")
         await vector_store.adelete_collection()
         del app.state.user_collections[user_id]
         log.info(f"Post Deletion {app.state.user_collections}")
         return JSONResponse(content={"message": f"Collection {collection_name} deleted Successfully"},status_code=200)
    except Exception:
         log.exception("Collection not deleted")
         return JSONResponse(content={"message": "Failed to delete collection"},status_code=500)
        
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

    