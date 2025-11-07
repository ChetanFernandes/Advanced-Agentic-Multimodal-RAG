import streamlit as st
import requests
import time
import logging as log

API_URL = "http://localhost:8000"


st.title("Advanced_RAG + Chat_GPT")
st.header("Upload File to Build a RAG System")


if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

if "available_sources" not in st.session_state:
    st.session_state.available_sources = []
    st.session_state.sources_loaded = False


def load_sources():
   response = requests.get(f"{API_URL}/available_sources")
   if response.status_code == 200:
        st.session_state.available_sources = response.json()["sources"]
   else:
        st.session_state.available_sources = []


# --- RUN ON FIRST APP LOAD ---
if not st.session_state.sources_loaded:
    load_sources()
    st.session_state.sources_loaded = True

if len(st.session_state.available_sources) == 0:
    st.warning("No documents found in vector DB yet. Upload files first!")


#File uploaded
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["xlsx", "docx", "pptx", "csv", "txt", "pdf"],
    key=f"upload_{st.session_state['uploader_key']}"
)

if uploaded_file is not None:
    if st.button("Upload File"):
        log.info("Uploading file to backend...")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            with st.spinner("Processing file..."):
                response = requests.post(f"{API_URL}/upload_file", files=files,timeout =300)

            st.write("Status:", response.status_code)
            st.write("Response:", response.text)

            if response.status_code == 200:
                st.success(response.json()["message"])
                load_sources() 
                time.sleep(2)
                st.session_state.uploader_key += 1  # new widget next time
                st.session_state["uploaded_file"] = None
                st.rerun()
            else:
                st.error("Failed to upload file.")

        except requests.exceptions.ConnectionError as e:
            log.exception(e)
            st.error("Backend crashed. Check FastAPI logs.")


# --- Query Interface ---
st.header("Multimodal Query Interface")
with st.form("query_form"):
    user_query = st.text_input("Enter your query")
    selected_doc = st.selectbox("Select document to query: ", st.session_state.available_sources,index=None)  # optionally get from API
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_query.strip():
        st.error("Please enter a question.")

    if uploaded_image:
        files = {'image': (uploaded_image.name, uploaded_image.getvalue(),uploaded_image.type)}
    else:
        files = {}

    data = {
            "query":user_query,
            "selected_doc" : selected_doc
           }
    with st.spinner("Processing query..."):
        response = requests.post(f"{API_URL}/query", data = data, files = files)

    st.write(response.json().get("result", "No result found") if response.status_code == 200 else "Query failed")


  




        





   

  




   

        
        
        

  
        





