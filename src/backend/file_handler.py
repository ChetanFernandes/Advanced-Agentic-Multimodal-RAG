import os
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain_community.document_loaders.json_loader import JSONLoader
import streamlit as st
from langchain.docstore.document import Document
from agentic_rag.backend.hybrid_pdf_parser import extract_pdf_elements
from agentic_rag.backend.hybrid_docx_parser import extract_docx_elements
from agentic_rag.backend.hybrid_excel_parser import extract_excel_elements
from agentic_rag.backend.hybrid_pptx_parser import extract_pptx_elements
import pandas as pd
import uuid
from agentic_rag.logger_config import log

def file_processor(file_name,file_bytes):
    file_type = file_name.rsplit(".", 1)[1].lower()
    print(file_type)

    if file_type == 'txt':
        if isinstance(file_bytes,bytes):
            content = file_bytes.decode('utf-8')
        documents = [Document(metadata={"source" : file_name}, page_content=content)]
    #doc = TextLoader(file,encoding = 'utf-8')
    
    elif file_type in ['xlsx', 'xls']:
        log.info("Entering under excel dunction")
        documents = extract_excel_elements(file_name)
    
    elif file_type == "pdf":
        log.info("Entering under pdf dunction")
        documents = extract_pdf_elements(file_name,file_bytes)

    elif file_type == "docx":
        log.info("Entering under doc dunction")
        documents = extract_docx_elements(file_name,file_bytes)

    elif file_type == "pptx":
        log.info("Entering under ppt dunction")
        documents = extract_pptx_elements(file_name)

    elif file_type =='csv':
        df = pd.read_csv(file_bytes)
        documents = [Document(page_content = row.to_json(), metadata = {"source": file_name, "row": id,"type":"csv"}) for id, row in df.iterrows()]

    elif file_type == "json":
        content = file_bytes.read()

         # Convert bytes to string if needed
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        loader = JSONLoader(file_path=None, jq_schema=".", text_content= True, json_string=content)
        raw_docs = loader.load()
        documents = [Document(page_content = doc.page_content, metadata = {"source": file_name, "id": str(uuid.uuid4()), "type":"json"}) for doc in raw_docs]

    else:
        # Fallback for doc, HTML, EML, EPUB, etc.
        documents = []
        loader = UnstructuredLoader(file_name, post_processors= [clean_extra_whitespace], strategy = "hi-res")
        raw_docs = loader.load()
        documents = [Document(page_content = doc.page_content, metadata = {"source": file_name, "id": str(uuid.uuid4()), "type":"text"}) for doc in raw_docs]
      
     
    return documents
