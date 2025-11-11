
from langchain.storage import InMemoryByteStore
from langchain_astradb import AstraDBVectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from models import *
from langchain.indexes import SQLRecordManager
import os
from dotenv import load_dotenv
from src.logger_config import log
import uuid


''' 
DataAPIClient → used to connect to Astra DB using the Data API.

VectorMetric → defines similarity metric (e.g., COSINE, DOT, EUCLIDEAN).

CollectionDefinition → defines how the collection should look (schema).

CollectionVectorOptions → defines vector-specific options (dimension, metric).
'''

class ConnectToAstraDB:
        def __init__(self):
            try:
                load_dotenv()
                self.api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
                self.app_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
                if not self.api_endpoint or not self.app_token:
                    raise ValueError("Astra DB API endpoint and token must be provided!")
            except Exception:
                 log.exception("AstraDB connection initialization failed")
                

        def initailize_vector_store(self,collection_name):
            """initialize a LangChain vector store for the given collection"""
            vector_store = AstraDBVectorStore(
                collection_name = collection_name,
                embedding = clip_embeddings,
                api_endpoint = self.api_endpoint,
                token = self.app_token,
                pre_delete_collection=False,
                metadata_indexing_include=["source", "type", "doc_id"]
            )
            return vector_store, vector_store.collection_name
    

        def get_multivector_retriever(self,collection_name):
            vector_store, collection_name = self.initailize_vector_store(collection_name)
            byte_store = InMemoryByteStore()
            id_key = "doc_id"  # or whatever you’re using
            log.info("Initialize the MultiVector retriever with the vector store")
            vector_retriever = MultiVectorRetriever(
                vectorstore=vector_store,
                docstore=byte_store,
                id_key=id_key,
                search_kwargs={"k": 3}
            )
            return vector_store, collection_name, vector_retriever
        
        def add_index(self,collection_name):
            """
            Adds indexing to the AstraDBVectorStore.
            Modes: 'full', 'incremental', or None
            """
            try:
                vector_store, collection_name,vector_retriever = self.get_multivector_retriever(collection_name)

                # Create a record manager (to keep track of indexed docs)
                namespace = f"AstraDBVectorStore/{collection_name}"
                record_manager = SQLRecordManager(
                    namespace,
                    db_url="sqlite:///astra_index_records.db"  # can also use postgres etc.
                )
                record_manager.create_schema()
                log.info(f"Vector store type: {type(vector_store)}")
                log.info(f"Retriever type: {type(vector_retriever)}")
                log.info(f"Record manager type: {type(record_manager)}")

                log.info("Adding index to AstraDB using the vector store and collection name")
                log.info("vector retriever and record manager/index were successfully created")
                return {
                "vector_store": vector_store,
                "collection_name": collection_name,
                "vector_retriever": vector_retriever,
                "record_manager": record_manager}
            except Exception:
                 log.exception("Failed to add index to AstraDB")
   

    

