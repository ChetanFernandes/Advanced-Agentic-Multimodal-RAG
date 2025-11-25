from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.backend.image_processing_bytes import extract_Image_summaries
from langchain_core.output_parsers import StrOutputParser
import asyncio
from langchain.indexes import index
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import chain
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever, ContextualCompressionRetriever
#from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank,Ranker
from langchain_community.document_compressors import FlashrankRerank
from src.logger_config import log



class Hybrid_retriever:
        def __init__(self,vector_store,vector_retriever,llm):
            try:
                self.vector_retriever = vector_retriever
                self.vector_store = vector_store
                #self.all_docs = self.vector_store.similarity_search("", k=1000)
                self.llm = llm
                log.info("✅ HybridRetriever initialized successfully.")
            except Exception:
                log.exception("Failed to initialize HybridRetriever")

        async def build(self,filter_metadata):
            ''' Build a hybrid retriever combining keyword, vector, and multi-query retrieval with compression '''
            try:
                all_docs = await self.vector_store.asimilarity_search("", k=1000)
                log.info("Using Ensemble retrievr to combine both keyword and vector retriever")
                if filter_metadata:
                    log.info(f"Filter metadata: {filter_metadata}")
                    # Apply to each retriever individually
                    if hasattr(self.vector_retriever, "search_kwargs"):
                        self.vector_retriever.search_kwargs.update({"filter": filter_metadata})

                keyword_retriever = None
                if filter_metadata and "source" in filter_metadata:
                    def filter_docs_by_source(docs, source_name):
                        return [doc for doc in docs if doc.metadata.get("source") == source_name]

                    filtered_docs = filter_docs_by_source(all_docs, filter_metadata["source"])
                    keyword_retriever = BM25Retriever.from_texts([doc.page_content for doc in filtered_docs])

                # Build ensemble
                hybrid_retriever = EnsembleRetriever(retrievers=[keyword_retriever, self.vector_retriever], weights=[0.5, 0.5])
                log.info("Ensemble retriever initialized successfully.")


                class LoggedMultiQueryRetriever(MultiQueryRetriever):
                    """Inner class that logs multiple queries."""
                    async def generate_queries(self, question: str, run_manager):
                        try:
                             if run_manager:
                                  queries = await super().agenerate_queries(question, run_manager=run_manager)
                             else:
                                  queries = await super().agenerate_queries(question)
                             log.info(f"MultiQueryRetriever generated {len(queries)} subqueries for: {question}")
                             for i, q in enumerate(queries, 1):
                                log.info(f"Generated subquery {i}: {q}")
                             return queries
                        except Exception:
                            log.exception("Failed to generate multiple queries")
                            return [question]  # fallback to single query

                    async def aget_relevant_documents(self, query: str, run_manager):
                        try:
                            queries = await self.generate_queries(query, run_manager=run_manager)

                            # Schedule parallel retrieval tasks
                            tasks = [self.retriever.aget_relevant_documents(q, run_manager=run_manager) for q in queries]

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

                        except Exception:
                            log.exception("Gennerating multiple queires failed")
                            return []

                retriever_from_llm = LoggedMultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm = self.llm)
        
                
                # Define async contextual compression retriever
                class AsyncContextualCompressionRetriever(ContextualCompressionRetriever):
                 
                    async def aget_relevant_documents(self, query, run_manager = None):
                        try:
                            # First, retrieve relevant docs asynchronously (already async)
                            docs = await self.base_retriever.aget_relevant_documents(query,run_manager = run_manager)
                            log.info("Docs retrived by Hybrid filter before re-rank")
                            log.info("\n" + "\n".join(
                                    [
                                    f"{'-' * 100}\n Document {i+1}:\n\n content:\n{doc.page_content}\n\n metadata:\n {doc.metadata}"
                                    if doc.page_content and doc.page_content.strip()
                                    else f"{'-'*100}\nDocument {i + 1}: No content"
                                    for i, doc in enumerate(docs)
                                    ]
                                ))

                            # Now compress them asynchronously (CPU offload)
                            compressed  = await asyncio.to_thread(self.base_compressor.compress_documents,docs,query)
                            return compressed
                        
                        except Exception:
                            log.exception("Failed during document compression")
                            return []
                compressor = FlashrankRerank()
                compression_retriever = AsyncContextualCompressionRetriever(base_compressor = compressor,base_retriever=retriever_from_llm)
                log.info("✅ Hybrid retriever pipeline setup complete")
                return compression_retriever
        
            except Exception:
                log.exception("Hybrid retriever pipeline setup failed.")
                return None



class question_answering:
    def __init__(self,llm,vector_store,vector_retriever,selected_doc):
        self.llm = llm
        self.vector_store = vector_store
        self.vector_retriever = vector_retriever
        self.selected_doc = selected_doc
        self.initilize_retriever = Hybrid_retriever(self.vector_store,self.vector_retriever,self.llm)
      
    
    async def retrieve_answer_from_query(self,query):
        
        compression_retriever = await self.initilize_retriever.build(filter_metadata={"source": self.selected_doc})

        log.info(f"Passing query to compression retriever: {query}")
        retrieved_docs = await compression_retriever.aget_relevant_documents(query) #Because now it’s async and can run directly on event loop (fast!).

        if not retrieved_docs:
            log.warning("⚠️ No relevant documents found.")
            return []

        log.info("Below is the results retrived")
        
        log.info("\n" + "\n".join(
                [
                 f"{'-' * 100}\n Document {i+1}:\n\n content:\n{doc.page_content}\n\n metadata:\n {doc.metadata}"
                 if doc.page_content and doc.page_content.strip()
                 else f"{'-'*100}\nDocument {i + 1}: No content"
                 for i, doc in enumerate(retrieved_docs)
                 ]
            ))
        return retrieved_docs

 
    async def extract_question_from_given_input(self,query,image):
        try:

            log.info(f"🟢 Received user text query: {query}")
            image_summary = []
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
            

            # Retrieve text context from RAG
            log.info("passing query to compression retriever")
            retrieved_docs = await self.retrieve_answer_from_query(query)
            if not retrieved_docs:
                log.info("No results retrived for given query")
                return [], query

            text_context = "\n".join([d.page_content for d in retrieved_docs])
            log.info(f"text context passed to LLM \n{text_context}")

            rag_found = bool(retrieved_docs)

            Final_context = ""
            
            log.info("Check what contet we are passing to LLM")
            if rag_found and image_summary:
                Final_context = f"Retrieved Text:\n{text_context}\n\nImage Summary:\n{image_summary}"
                log.info(f"Passing both text and image context to LLM - \n {Final_context}")
                system_msg = (
                    "You are a helpful assistant. Use both the retrieved text context "
                    "and the image summary to answer the question accurately."
                )

            elif rag_found:
                Final_context = f"Retrieved Text:\n{text_context}"
                log.info(f"Passing only text context to LLM - \n {Final_context}")
                system_msg = (
                    "You are a helpful assistant. Use the retrieved text context to answer the question."
                )

            elif image_summary:
                Final_context = f"Image Summary:\n{image_summary}"
                log.info(f"Passing only image summary to LLM \n {Final_context}")
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
        
            result = await chain.ainvoke({
                "question": query,
                "context": Final_context,
            })
            if not result:
                log.info("No results from LLM")
                return result, query

            log.info("Final Answer from LLM:")
            log.info(f"{result}")

            return result,query
    
        except Exception:
            log.exception("Result extraction failed")
            return [], query


    
    




        

        

        

        





    

    

    

    
        







    













    



    




    






