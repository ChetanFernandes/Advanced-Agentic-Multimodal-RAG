import streamlit as st
import requests
import time
from logger_config import log
import jwt
import extra_streamlit_components as stx


cookie_manager = stx.CookieManager()

API_URL = "http://localhost:8000"
JWT_SECRET = st.secrets["auth"]["JWT_SECRET"]
JWT_ALGO = "HS256"


st.set_page_config(page_title="RAG App Login", page_icon="🔐")

# 1. First: Try to load user from cookies
if not st.session_state.get("logging_out"):
    user_cookie = cookie_manager.get("user")
    token_cookie = cookie_manager.get("jwt_token")
else:
    user_cookie = None
    token_cookie = None

if "logging_out" in st.session_state:
    st.session_state.pop("logging_out")

if user_cookie and token_cookie:
    st.session_state["user"] = user_cookie
    st.session_state["jwt_token"] = token_cookie


# 2. Next: Try OAuth callback ONLY IF no user found in session_state
# -------------------------------------------------------------
if "user" not in st.session_state or st.session_state["user"] is None:
        
        # Read the token returned from FastAPI callback
        query_params = st.query_params
        token_list = query_params.get("token")
        log.info(f" Token List -> {token_list}")
        
        if token_list:
            jwt_token = token_list # extract token from url POST hack

            try:
                decoded = jwt.decode(jwt_token, JWT_SECRET, algorithms=[JWT_ALGO])
                user_info = decoded["user"]

                # Save login info in session
                st.session_state["user"] = user_info
                st.session_state["jwt_token"] = jwt_token

                cookie_manager.set("user", user_info, key="cookie_user_setter")
                cookie_manager.set("jwt_token", jwt_token, key="cookie_jwt_setter")

                st.query_params.clear()

        
            except jwt.ExpiredSignatureError:
                st.error("Login expired! Please log in again.")
                st.stop()

            except jwt.InvalidTokenError:
                st.error("Invalid login token.")
                st.stop()



# 3. If still no user → show login screen
# -------------------------------------------------------------
if "user" not in st.session_state or st.session_state["user"] is None:
    st.subheader("Please log in to continue")
    st.markdown(f'<a href="{API_URL}/login" target="_self">👉 Log in with Google</a>', unsafe_allow_html=True)
    st.stop()
# ---------------------------


# 4. User is logged in from cookies or callback → continue
# ---------------------------
user = st.session_state["user"]
jwt_token = st.session_state["jwt_token"]

st.success(f"Welcome {user['name']} 👋 ({user['email']})")
user_id = user["sub"]

if  st.button("Logout"):
    response = requests.post(f"{API_URL}/logout", headers = st.session_state.get("auth_headers"))
    if response.status_code == 200:
        st.write(response.json().get("message"))
        st.session_state["logging_out"] = True 
        st.session_state.pop("user", None)
        st.session_state.pop("jwt_token", None)
        st.session_state.pop("auth_headers", None)
        st.cache_data.clear()
        cookie_manager.delete("user", key="cookie_user_setter")
        cookie_manager.delete("jwt_token", key="cookie_jwt_setter")
        #st.session_state.pop("available_sources", None)
        st.rerun()
    else:
        st.session_state["logging_out"] = True
        st.session_state.clear()
        st.rerun()


if "auth_headers" not in st.session_state:
    st.session_state.auth_headers = {"Authorization": f"Bearer {jwt_token}"}
else:
    st.session_state.auth_headers["Authorization"] = f"Bearer {jwt_token}"

#st.write("AUTH HEADERS:", st.session_state.get("auth_headers"))


if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0


if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

@st.cache_data(ttl=3600)
def get_sources():
    """Fetches available document sources from backend."""
    try:
        #st.write("Entering load_source function")
        response = requests.get(f"{API_URL}/available_sources", headers = st.session_state.get("auth_headers"))
        #st.write(f"🔍 Response status: {response.status_code}")
        #st.write(f"🔍 Raw response: {response.text}")

        if response.status_code == 200:
            st.session_state.available_sources = response.json().get("sources", [])
            log.info("available_sources")
            return st.session_state.available_sources
        else:
            log.info("No available resources found")
            return []

    except Exception as e:
        log.info("Entering except function")
        log.exception(f"Failed to load sources: {e}")
        return []


def load_sources(force_refresh=False):
    """Fetch sources with optional cache clear."""
    if force_refresh:
        get_sources.clear()
    st.session_state.available_sources = get_sources()
    return st.session_state.available_sources

    #if not st.session_state.available_sources:
        #st.session_state.pop("selected_doc", None)
    

if "available_sources" not in st.session_state:
    st.write("Resouce not in session state")
    st.session_state.available_sources = load_sources()
#st.session_state.available_sources = load_sources()
#else:
    #st.write("Resouce in session state")
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
        try:
            log.info('Enter upload file function')
            with st.spinner("Processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/upload_file", headers = st.session_state["auth_headers"], files=files)
                if response.status_code == 200:
                    st.success(response.json().get("message"))
                    time.sleep(1.5)
                    load_sources(force_refresh=True)
                    st.session_state.uploader_key += 1  # new widget next time
                    st.session_state["uploaded_file"] = None
                    st.rerun()
                else:
                    st.error(response.json().get("message"))
        except Exception:
            log.exception("File upload failed")




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
        st.stop()

    if uploaded_image:
        files = {'image': (uploaded_image.name, uploaded_image.getvalue(),uploaded_image.type)}
    else:
        files = {}

    data = {
            "query":user_query,
            "selected_doc" : selected_doc 
        }
    with st.spinner("Processing query..."):
        response = requests.post(f"{API_URL}/query", data = data, files = files, headers = st.session_state["auth_headers"])

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
                    st.session_state.pop("selected_doc", None)
                    st.rerun()
                else:
                    result = response.json().get("message", "Failed to delete collection.")
                    st.error(result)

else:
    st.info("No collection in DB exists. Upload your documents to use RAG.")
  









   

  




   

        
        
        

  
        





