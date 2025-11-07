import logging as log
import uuid
from agentic_rag.backend.file_handler import file_processor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import index
import streamlit as st
from agentic_rag.logger_config import log


class Adding_files_DB:
    def __init__(self,vector_retriever,record_manager,vector_store,file_name,file_bytes):
        self.record_manager = record_manager
        self.vector_store = vector_store
        self.vector_retriever = vector_retriever
        self.file_name = file_name
        self.file_bytes = file_bytes
 
    def in_memory_store(self):
        documents = file_processor(self.file_name,self.file_bytes)
        if documents:
            log.info(
                    "\n" + "\n".join(
                        [
                            f"{'-'*100}\nDocument {i + 1}:\n\nContent:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}"
                            if doc.page_content and doc.page_content.strip()
                            else f"{'-'*100}\nDocument {i + 1}: No content"
                            for i, doc in enumerate(documents)
                        ]
                        )
                        )
        else:
            log.info("No documents found")
  
        id_key = "doc_id"
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        all_child_docs = []
   
        for i, parent_doc in enumerate(documents):
            _id = doc_ids[i]
            splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " ", ""], chunk_size = 500, chunk_overlap = 100)
            child_docs = splitter.split_documents([parent_doc])
            log.info(f"parent documents splitted in to child chunks. Lenghth of splitted doc is {len(child_docs)}")

            # assign parent id to child chunks
            log.info(f"Entering loop to assign parent id to child chunks")
            for doc in child_docs:
                doc.metadata[id_key] = _id

            all_child_docs.extend(child_docs)
            log.info(f"Mapped child chunks to their respective parent ID")

        log.info("Using index to store the child chunks in Astra DB, to avaoid duplicates")
        index_instance = index(
                        all_child_docs,
                        self.record_manager ,
                        self.vector_store,
                        cleanup="incremental",
                        source_id_key="source"
                   )
        log.info("child_chunks uploaded in ASTRA Database")
        self.vector_retriever.docstore.mset(list(zip(doc_ids, documents)))
        log.info("Added parent document to vector retriever docstore")

