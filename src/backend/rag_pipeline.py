import asyncio
import re
import logging as log
from utility import Hybrid_retriever, question_answering, extract_Image_summaries, Independent_image_upload



if submitted:
    if not user_query.strip():
        st.error("🚨 Please enter a question.")

    elif not selected_doc:
        log.info("Document not selected. Pass query directly to agent on hittig submit button")
        query = { "question" : user_query,
                   "image" : uploaded_image}
         
        text_query = query.get("question")
        image_query = query.get("image")
        image_context = ""

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
            image_context = asyncio.run(extract_Image_summaries(save_path)) if save_path else ""
            log.info(f"Image summary {image_context}")

        log.info("Passing text query and image summary to agent for final results")
        final_response = web_search_agent.query_answering(agent,text_query,image_context)
        final_response = final_response.get("output", final_response)
        log.info(f"Final Answer\n, {final_response}")
        st.write(final_response)

    else:
        st.success("✅ Submitted successfully!")
        query = { "question" : user_query,
                "image" : uploaded_image}
       
        log.info("Entering function to create a hybrid + compression retriever ")
        
        initilize_retriever = Hybrid_retriever(vector_store,vector_retriever,llm)

        compression_retriever = initilize_retriever.build(filter_metadata={"source": selected_doc})

        log.info(f"compressed retriver {compression_retriever}")


        log.info("Entering function for answering given query")
        answer = question_answering(llm,compression_retriever,agent,web_search_agent)

        retrived_results,text_query  = answer.extract_question_from_given_input(query)

        log.info("Passing retrived results and query to agent for final results")
        final_response = web_search_agent.query_answering(agent,text_query,retrived_results)
        final_response = final_response.get("output", final_response)
        log.info(f"Final Answer\n, {final_response}")
        st.write(final_response)

        # Wait before clearing
        #time.sleep(4)

        # --- Reset all inputs ---
        #st.rerun()
        


log.info("\n[DEBUG] Memory so far:")
for idx, msg in enumerate(memory.chat_memory.messages, 1):
    sender = "USER" if msg.type == "human" else "ASSISTANT"
    log.info(f"{idx}. [{sender}]: {msg.content}")

       