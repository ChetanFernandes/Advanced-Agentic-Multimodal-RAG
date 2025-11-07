# FastAPI is built on top of asyncio, a Python library that supports asynchronous programming — 
# meaning tasks can run without blocking each other.

from fastapi import FastAPI, UploadFile, File, Form 
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager
from agentic_rag.backend.DB import ConnectToAstraDB
from agentic_rag.backend.agent import web_agent
from agentic_rag.backend.Adding_files import Adding_files_DB
from agentic_rag.backend.utilis import *
from agentic_rag.backend.image_processing_bytes import extract_Image_summaries
import asyncio
import logging as log
import os
from langchain_community.chat_models import ChatOllama
from fastapi.responses import JSONResponse
from agentic_rag.backend.chunking_retrieveing import Hybrid_retriever, question_answering
from fastapi.responses import JSONResponse
from fastapi import Request
from agentic_rag.logger_config import log


app = FastAPI()


# Allow Streamlit frontend to call this API
app.add_middleware(CORSMiddleware,
                   allow_origins = ["*"],
                   allow_credentials = True,
                   allow_methods = ["*"],
                   allow_headers = ["*"],
                   )


@asynccontextmanager # Tells - “Hey, when the app starts, run this function once before handling any user requests.”
async def lifespan(app: FastAPI):
    log.info("Server is starting up...")

    app.state.ASTRA_DB = ConnectToAstraDB()
    app.state.vector_store, app.state.collection_name, app.state.vector_retriever,app.state.record_manager  = await asyncio.to_thread(app.state.ASTRA_DB.add_index)

    app.state.llm = ChatOllama(model="qwen2.5vl:3b")


    app.state.web_search_agent = web_agent(app.state.llm)
    app.state.agent = await asyncio.to_thread(app.state.web_search_agent.initializing_agent)
    app.state.memory =  app.state.web_search_agent.memory

    
    log.info("Startup initialization complete")

    # Yield control back to FastAPI (server runs after this)
    yield #→ tells FastAPI “do the startup code before yield, then run the server; after the server stops, run shutdown code after yield”

    # SHUTDOWN (runs once when app stops)
    log.info("Shutting down gracefully...")
    # Optionally: close DB connections or cleanup models
    
app.router.lifespan_context = lifespan
# router is the internal object FastAPI uses to manage all endpoints (routes).

# Add an endpoint in FastAPI (backend)
@app.get("/available_sources")
async def available_sources():
        try:
            # Fetch all docs from vector store
            all_docs = await app.state.vector_store.asimilarity_search("", k=1000)
            app.state.available_doc_names = list({doc.metadata.get("source") for doc in all_docs if doc.metadata.get("source")})
            log.info(f"available_doc_names -> {app.state.available_doc_names}")
            return JSONResponse(
                        content={"sources": app.state.available_doc_names},
                        headers = {"X-Owner":"Chetan"},
                        status_code = 200
                    )
        except Exception as e:
            log.exception(e)
            return JSONResponse(
                content={"message": f"Failed to get available sources: {str(e)}"},
                status_code=500
            )

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
        data = await file.read()
        file.file.seek(0)
        log.info(f"Uploading file: {file.filename}, size={len(data)}")

        file_name = file.filename
        file_bytes = data  # Now safe!
   
    
        vector_retriever = app.state.vector_retriever
        record_manager = app.state.record_manager
        vector_store = app.state.vector_store
        
        DB_process = Adding_files_DB(vector_retriever,record_manager,vector_store,file_name,file_bytes)
    
        await asyncio.to_thread(DB_process.in_memory_store)

        log.info(f"File '{file_name}' processed successfully.")

        return JSONResponse(
                content={"message": f" File {file_name} uploaded and processed successfully."},
                status_code=200
            )

    except Exception as e:
            # If anything goes wrong, return an error
            log.exception(e)
            return JSONResponse(
                content={"message": f"Failed to process file: {str(e)}"},
                status_code=500
            )
    

@app.post("/query")
async def query_endpoint(query:str = Form(...), selected_doc:str = Form(None), image:  UploadFile = File(None)):
    try:
        agent = app.state.agent,
        web_search_agent = app.state.web_search_agent,
        vector_retriver = app.state.vector_retriver,
        vector_store = app.state.vector_store,

        llm = app.state.llm

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
                image_summary = "\n".join(image_summary) if isinstance(image_summary, list) else str(image_summary)
        
            log.info("Passing text query and image summary to agent for final results")
            if hasattr(web_search_agent, "query_answering_async"):
                final_response = await web_search_agent.query_answering_async(agent,query,image_summary)
            else:
                log.error("No query_answering_async exists ")
        
            final_response = final_response.get("output", final_response)
            log.info(f"Final Answer\n, {final_response}")
            return JSONResponse(
                        content={"result": f"{final_response}"},
                        status_code=200
        )
        
        else:
            log.info("Entering function to create a hybrid + compression retriever")
        
            initilize_retriever = Hybrid_retriever(vector_store,vector_retriver,llm)

            compression_retriever = await asyncio.to_thread(initilize_retriever.build(filter_metadata={"source": selected_doc}))

            log.info(f"compressed retriver {compression_retriever}")
            log.info("Entering function for answering given query")

            answer = question_answering(llm,compression_retriever,agent,web_search_agent)

            retrived_results,text_query  = await answer.extract_question_from_given_input(query)

            log.info("Passing retrived results and query to agent for final results")
            final_response = await web_search_agent.query_answering_async(agent,text_query,retrived_results)
            
            if isinstance(final_response, dict) and "output" in final_response:
                 final_response = final_response["output"]

            log.info(f"Final Answer\n, {final_response}")
            return JSONResponse(
                        content={"result": f"{final_response}"},
                        status_code=200)

    except Exception as e:
            log.exception(e)
            return JSONResponse(
                content={"message": f"Failed to answer question"},
                status_code=500
            )

    
    
     

          



          
     
     
    







  
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

