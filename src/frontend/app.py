import streamlit as st
import requests
import time
from logger_config import logging as log
import uuid
from streamlit_cookies_controller import CookieController
import datetime
API_URL = "http://localhost:8000"

# Try to get 'uid' from the URL
query_uid = st.query_params.get("uid", [None])[0]

if "user_id" not in st.session_state:
    if query_uid:
        # Use URL if present on first page load
        st.session_state.user_id = query_uid
    else:
        # No URL param, generate new
        new_id = uuid.uuid4().hex[:8]
        st.session_state.user_id = new_id
        # Set param and force reload
        st.query_params["uid"] = new_id
        st.rerun()  # This halts execution here!
        
# Always use session_state after its initialized
user_id = st.session_state.user_id

st.write(f"User ID: {user_id}")

st.write(f"✅ Your persistent user ID: {user_id}")

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0


if "available_sources" not in st.session_state:
    st.session_state.available_sources = []
    #st.session_state.sources_loaded = False

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

@st.cache_data(ttl=3600)
def get_sources(user_id):
    """Fetches available document sources from backend."""
    try:
        #st.write("Entering load_source function")
        response = requests.get(f"{API_URL}/available_sources", params={'user_id': user_id})
        #st.write(f"🔍 Response status: {response.status_code}")
        #st.write(f"🔍 Raw response: {response.text}")

        if response.status_code == 200:
            st.session_state.available_sources = response.json().get("sources", [])
            log.info("available_sources")
            return st.session_state.available_sources
        else:
            log.info("Entering else function")
            return []

    except Exception as e:
        log.info("Entering except function")
        log.exception(f"Failed to load sources: {e}")
        return []


def load_sources(force_refresh=False):
    """Fetch sources with optional cache clear."""
    if force_refresh:
        get_sources.clear()
    st.session_state.available_sources = get_sources(user_id)


    if not st.session_state.available_sources:
        st.session_state.pop("selected_doc", None)
    return st.session_state.available_sources


# Normal refresh (fast, cached)
st.session_state.available_sources = load_sources()
#st.write(st.session_state.available_sources)


# --- UI HEADER --
st.title("Advanced_RAG + Chat_GPT")
st.header("Upload File to Build a RAG System")


#File uploaded
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["xlsx", "docx", "pptx", "csv", "txt", "pdf"],
    key=f"upload_{st.session_state['uploader_key']}"
)

if uploaded_file is not None and st.button("Upload File"):
        with st.spinner("Processing file..."):
         files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
         response = requests.post(f"{API_URL}/upload_file", params = {"user_id" : user_id}, files=files,timeout =300)

         if response.status_code == 200:
            st.success(response.json().get("message"))
            time.sleep(1.5)
            load_sources(force_refresh=True)
            st.session_state.uploader_key += 1  # new widget next time
            st.session_state["uploaded_file"] = None
            st.rerun()
         else:
            st.error(response.json().get("message"))




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
        response = requests.post(f"{API_URL}/query", data = data, files = files, params = {"user_id" : user_id})

    if response.status_code == 200:
        result = response.json()
        st.subheader("🧠 Final Answer")
        st.markdown(result.get("result", "No result found."))
    else:
        response.json().get("status", "Failed to answer question (Internal_error)")

 
    #st.write(response.json().get("result", "No result found") if response.status_code == 200 else response.json().get("status", "Failed to answer question (Internal_error)"))


st.header("🗑️ Manage Collection")

if st.session_state.available_sources:
    if st.button("Delete your document uploaded in DB"):
        with st.spinner("Deleting your collection..."):
                response = requests.delete(f"{API_URL}/delete_collection",params={"user_id": user_id},timeout=120)
                if response.status_code == 200:
                    st.success(response.json().get("message", "Collection deleted successfully."))
                    load_sources(force_refresh=True)
                    st.rerun()
                else:
                    result = response.json().get("message", "Failed to delete collection.")
                    st.error(result)

else:
    st.info("No collection in DB exists. Upload your documents to use RAG.")
  









   

  




   

        
        
        

  
        





