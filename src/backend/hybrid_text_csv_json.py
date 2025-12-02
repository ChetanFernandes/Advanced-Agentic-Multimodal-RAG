
from langchain.docstore.document import Document
from src.backend.utilis import *
import uuid
from src.logger_config import log
import pandas as pd
from langchain_community.document_loaders.json_loader import JSONLoader 
from io import BytesIO


def txt_file_processing(file_name,file_bytes):
    try:
        if isinstance(file_bytes,bytes):
            content = file_bytes.decode('utf-8')
        documents  = [Document(metadata={"source" : file_name, "type": "txt"}, page_content=content)]  #doc = TextLoader(file,encoding = 'utf-8')
        if not documents:
            log.warning(f"Text parsed but no valid text documents found in {file_name}")
            return [], "No readable text detected."

        log.info(f"Extracted {len(documents)} document chunks from {file_name}")
        return documents, None  # No error message

    except Exception as e:
        log.exception("Creating documents object failed")
        return [], f"Text extraction failed: {e}"
    
def csv_file_processing(file_name,file_bytes):
        try:
            df = pd.read_csv(BytesIO(file_bytes))
            documents = [Document(page_content = row.to_json(), metadata = {"source": file_name, "row": id,"type":"csv"}) for id, row in df.iterrows()]
            if not documents:
                log.warning(f"Text parsed but no valid text documents found in {file_name}")
                return [], "No readable text detected."

            log.info(f"Extracted {len(documents)} document chunks from {file_name}")
            return documents, None  # No error message
        except Exception as e:
            log.exception("Creating documents object failed")
            return [], f"CSV  extraction failed: {e}"
        
     
    
def JSON_file_processing(file_name,file_bytes):
    try:
        content = file_bytes.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        loader = JSONLoader(file_path=None, jq_schema=".", text_content= True, json_string=content)
        raw_docs = loader.load()
        documents  = [Document(page_content = doc.page_content, metadata = {"source": file_name, "id": str(uuid.uuid4()), "type":"json"}) for doc in raw_docs]
        if not documents:
                log.warning(f"Text parsed but no valid text documents found in {file_name}")
                return [], "No readable text detected."

        log.info(f"Extracted {len(documents)} document chunks from {file_name}")
        return documents, None  # No error message
    except Exception as e:
        log.exception("Creating documents object failed")
        return [], f"json  extraction failed: {e}"
    


''' 
        # Fallback for doc, HTML, EML, EPUB, etc.
        documents = []
        loader = UnstructuredLoader(file_name, post_processors= [clean_extra_whitespace], strategy = "hi-res")
        raw_docs = loader.load()
        documents = [Document(page_content = doc.page_content, metadata = {"source": file_name, "id": str(uuid.uuid4()), "type":"text"}) for doc in raw_docs]
'''