from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from agentic_rag.backend.image_processing_bytes import extract_Image_summaries
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import asyncio
import streamlit as st
from langchain.indexes import index
from agentic_rag.models import *
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain_core.runnables import chain
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from agentic_rag.logger_config import log
import os
from agentic_rag.backend.utilis import Independent_image_upload
import re


class Hybrid_retriever:
        def __init__(self,vector_store,vector_retriever,llm):
            self.vector_retriever = vector_retriever
            self.vector_store = vector_store
            self.all_docs = self.vector_store.similarity_search(" ", k=1000)
            self.llm = llm

 
        def build(self,filter_metadata):
            log.info("Using Ensemble retrievr to combine both keyword and vector retriever")

            if filter_metadata:
                log.info(f"Filter metadata: {filter_metadata}")
                # Apply to each retriever individually
                if hasattr(self.vector_retriever, "search_kwargs"):
                    self.vector_retriever.search_kwargs.update({"filter": filter_metadata})

            if filter_metadata and "source" in filter_metadata:

                def filter_docs_by_source(docs, source_name):
                     return [doc for doc in docs if doc.metadata.get("source") == source_name]

                filtered_docs = filter_docs_by_source(self.all_docs, filter_metadata["source"])
                keyword_retriever = BM25Retriever.from_texts([doc.page_content for doc in filtered_docs])

            # Build ensemble
            hybrid_retriever = EnsembleRetriever(retrievers=[keyword_retriever, self.vector_retriever], weights=[0.5, 0.5])
            log.info("Using Multiquery retriver to genereate more queries")

  
            class LoggedMultiQueryRetriever(MultiQueryRetriever):
                """Inner class that logs multiple queries."""

                async def generate_queries(self, question: str, run_manager=None):
                    queries = await super().agenerate_queries(question, run_manager=run_manager)
                    log.info(f"MultiQueryRetriever generated {len(queries)} subqueries for: {question}")
                    for i, q in enumerate(queries, 1):
                        log.info(f"Generated subquery {i}: {q}")
                    return queries

                async def _aget_relevant_documents(self, query: str, run_manager=None):
                    # Generate all sub-queries
                    queries = await self.generate_queries(query, run_manager=run_manager)

                    # Schedule parallel retrieval tasks
                    tasks = [
                        self._base_retriever.aget_relevant_documents(q)
                        for q in queries
                    ]

                    # Run all retrievals concurrently
                    results = await asyncio.gather(*tasks)

                    # Flatten List[List[Document]] → List[Document]
                    merged = [doc for sublist in results for doc in sublist]

                    # Optional deduplication
                    seen = set()
                    unique_docs = []
                    for doc in merged:
                        key = hash(doc.page_content)
                        if key not in seen:
                            seen.add(key)
                            unique_docs.append(doc)

                    return unique_docs
                
            retriever_from_llm = LoggedMultiQueryRetriever.from_llm(hybrid_retriever, llm = self.llm)
                
            class AsyncContextualCompressionRetriever(ContextualCompressionRetriever):

                async def aget_relevant_documents(self, query, run_manager=None):
                    # First, retrieve relevant docs asynchronously (already async)
                    docs = await self.base_retriever._aget_relevant_documents(query)

                    # Now compress them asynchronously (CPU offload)
                    compressed = await asyncio.to_thread(self.base_compressor.compress_documents,docs,query)

                    return compressed


            log.info("using flash rank to rerank the documents retrieved and compress")
            #compressor = FlashrankRerank()
            compression_retriever = AsyncContextualCompressionRetriever(base_compressor = FlashrankRerank(),base_retriever=retriever_from_llm)
            log.info("✅ Hybrid retriever pipeline setup complete")
            return compression_retriever
            

class question_answering:
    def __init__(self,llm,compression_retriever,agent,web_search_agent):
        self.compression_retriever = compression_retriever
        self.llm = llm
        self.agent = agent
        self.instance_web_search_agent = web_search_agent
    
    async def retrieve_answer_from_query(self,query):

        log.info(f"🟠 Passing query to compression retriever: {query}")
        results = await self.compression_retriever.ainvoke(query) #Because now it’s async and can run directly on event loop (fast!).

        if not results:
            log.warning("⚠️ No relevant documents found.")
            st.warning("No relevant documents found")

        log.info("Below is the results retrived")
        
        log.info("\n" + "\n".join(
                [
                 f"{'-' * 100}\n Document {i+1}:\n\n content:\n{doc.page_content}\n\n metadata:\n {doc.metadata}"
                 if doc.page_content and doc.page_content.strip()
                 else f"{'-'*100}\nDocument {i + 1}: No content"
                 for i, doc in enumerate(results)
                 ]
            ))
        return results

 
    async def extract_question_from_given_input(self,query):
        text_query = query.get("question")
        image_query = query.get("image")
        image_context = ""

        log.info(f"🟢 Received user text query: {text_query}")

        if image_query:
            log.info("Store the uploaded image in folder")
            image_bytes = image_query.read()
            path_image = Independent_image_upload()
            safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', image_query.name)
            save_path = os.path.join(path_image, safe_filename)
            with open(save_path, "wb") as f:
                f.write(image_bytes)

            log.info(f"Image stored folder ->> {save_path}")
            log.info("pass the image to get its summary")
            image_context = await extract_Image_summaries(save_path) if save_path else ""
            log.info(f"Image summary {image_context}")

        
        # Retrieve text context from RAG
        log.info("passing query to compression retriever")
        results = await self.retrieve_answer_from_query(text_query) if text_query else []

        text_context = "\n".join([d.page_content for d in results])
        log.info(f"text context passed to LLM \n{text_context}")

        rag_found = bool(results)

        combined_context = ""
        
        log.info("Check what contet we are passing to LLM")
        if rag_found and image_context:
            combined_context = f"Retrieved Text:\n{text_context}\n\nImage Summary:\n{image_context}"
            log.info(f"Passing both text and image context to LLM - \n {combined_context}")
            system_msg = (
                "You are a helpful assistant. Use both the retrieved text context "
                "and the image summary to answer the question accurately."
            )

        elif rag_found:
            combined_context = f"Retrieved Text:\n{text_context}"
            log.info(f"Passing only text context to LLM - \n {combined_context}")
            system_msg = (
                "You are a helpful assistant. Use the retrieved text context to answer the question."
            )

        elif image_context:
            combined_context = f"Image Summary:\n{image_context}"
            log.info(f"Passing only image summary to LLM \n {combined_context}")
            system_msg = (
                "No text context found. Answer based on the image summary. Indicate this is visual analysis"
            )
        else:
            log.info("No results dervived from LLM")
            system_msg = (
                "No context available. Please upload a document or image."
            )
        

        # Start with system + user question
        prompt = ChatPromptTemplate.from_messages([
        (SystemMessagePromptTemplate.from_template(system_msg)),
        (HumanMessagePromptTemplate.from_template("Question: {question}\nContext: {context}"))])

        
        log.info("Invoke the chain")
        # Invoke with actual user input
        chain = prompt | self.llm  | StrOutputParser()
      
        retrived_results = await chain.ainvoke({
            "question": text_query,
            "context": combined_context,
        })
        log.info("💬 Final Answer from LLM:")
        log.info(f"{retrived_results}")
        return retrived_results,text_query

        # Retrieve Answer First
        # Sends the query to the qa_chain.
        # The retriever pulls relevant documents.
        # The LLM summarizes them.
        # "result" holds the actual answer
        
        # Let the agent decide whether to use tools/memory

    
    




        

        

        

        





    

    

    

    
        







    













    



    




    






