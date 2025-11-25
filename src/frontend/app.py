import streamlit as st
import requests
import time
import jwt, sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger_config import log
import extra_streamlit_components as stx

st.set_page_config(page_title="RAG App Login", page_icon="🔐")
st.markdown("<h2>⚡🧠 Advanced Agentic RAG + ChatGPT</h3>", unsafe_allow_html=True)

# -------------------- BACKEND BASE URL --------------------
# In Docker (AWS): BACKEND_URL=/api   → calls like /api/health
# Locally: BACKEND_URL not set       → http://localhost:8000
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
log.info(f"Using Backend URL: {API_URL}")


# ---- BACKEND HEALTH CHECK ----
healthy = False
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        health_data = health.json()
        if health_data.get("success") == "Backend is healthy":
            healthy = True
except:
    pass

if not healthy:
    st.warning("⛔ Backend not ready yet… UI will load but features may not work.")


JWT_SECRET = st.secrets["JWT_SECRET"]
JWT_ALGO = "HS256"
cookie_manager = stx.CookieManager()


# Initialize flags

st.session_state.setdefault("user", None)
st.session_state.setdefault("jwt_token", None)
st.session_state.setdefault("auth_headers", None)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("sources_loaded", False)
st.session_state.setdefault("available_sources", [])
st.session_state.setdefault("uploaded_file", [])
st.session_state.setdefault("selected_doc", [])
st.session_state.setdefault("logging_out", False)
st.session_state.setdefault("did_oauth", False)



if not st.session_state.get("logging_out"): # If False, then run this block.
    #st.write("Eroror")
    user_cookie = cookie_manager.get("user")
    token_cookie = cookie_manager.get("jwt_token")
else:
    user_cookie = None
    token_cookie = None


if user_cookie and token_cookie:
    #st.write("Eroror")
    st.session_state["user"] = user_cookie
    st.session_state["jwt_token"] = token_cookie


# 2. Next: Try OAuth callback ONLY IF no user found in session_state
# -------------------------------------------------------------
if (
    not st.session_state.get("user") 
    and not st.session_state.get("jwt_token")
    and not st.session_state.get("logging_out")
):
        # Read the token returned from FastAPI callback
        #st.write("Eroror")
        query_params = st.query_params
        token_list = query_params.get("token")
    

        if token_list and not st.session_state["did_oauth"]:  
            jwt_token = token_list # extract token from url POST hack

            try:
                decoded = jwt.decode(jwt_token, JWT_SECRET, algorithms=[JWT_ALGO])
                user_info = decoded["user"]

                # Save login info in session
                st.session_state["user"] = user_info
                st.session_state["jwt_token"] = jwt_token


                #Prevent repeat OAuth handling
                st.session_state["did_oauth"] = True 

                # Store cookies
                cookie_manager.set("user", user_info, key="cookie_user_setter")
                cookie_manager.set("jwt_token", jwt_token, key="cookie_jwt_setter")
    
                # Prevent repeat OAuth handling
                st.query_params.clear()

        
            except jwt.ExpiredSignatureError:
                st.error("Login expired! Please log in again.")
                st.stop()

            except jwt.InvalidTokenError:
                st.error("Invalid login token.")
                st.stop()

# 3️⃣ If no user yet → show Login screen
if "user" not in st.session_state or st.session_state["user"] is None:
    # If we came here after logout, clear the flag now
    if "logging_out" in st.session_state:
        st.session_state.pop("logging_out")

    st.subheader("Please log in to continue")

    # login URL:
    # - locally: API_URL = "http://localhost:8000"  → "http://localhost:8000/login"
    # - docker:  API_URL = "/api"                   → "/api/login"
    login_url = f"{API_URL}/login"

    st.markdown(
        f'<a href="{login_url}" target="_self">👉 Log in with Google</a>',
        unsafe_allow_html=True,
    )
    st.stop()
# ---------------------------

# 4. User is logged in from cookies or callback → continue
# ---------------------------
user = st.session_state["user"]
jwt_token = st.session_state["jwt_token"]
st.success(f"Welcome {user['name']} 👋 ({user['email']})")
st.session_state.auth_headers = {"Authorization": f"Bearer {jwt_token}"}
user_id = user["sub"]

@st.cache_data(ttl=3600)
def get_sources():
    """Fetches available document sources from backend."""
    try:
        #st.write("Entering load_source function")
        response = requests.get(f"{API_URL}/available_sources", headers = st.session_state.get("auth_headers"))
        #st.write(f"🔍 Response status: {response.status_code}")
        #st.write(f"🔍 Raw response: {response.text}")

        if response.status_code == 200:
            return response.json().get("sources", [])
        else:
            log.info("No available resources found")
            return []

    except Exception as e:
        log.info("Entering except function")
        log.exception(f"Failed to load sources: {e}")
        return []


if not st.session_state["sources_loaded"]:
    st.session_state["available_sources"] = get_sources()
    st.session_state["sources_loaded"] = True


# --- UI HEADER --


st.markdown("<h3>📁 Upload File to Build Your RAG System</h3>", unsafe_allow_html=True)


#File uploaded
uploaded_file = st.file_uploader(
    "Choose file",
    type=["xlsx", "docx", "pptx", "csv", "txt", "pdf"],
    key=f"upload_{st.session_state['uploader_key']}"
)

if uploaded_file is not None and st.button("Upload File"):
        try:
            log.info('Enter upload file function')
            with st.spinner("Processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/upload_file",files=files, headers = st.session_state["auth_headers"])
                if response.status_code == 200:
                    st.success(response.json().get("message"))
                    get_sources.clear()   
                    st.session_state["sources_loaded"] = False
                    st.session_state.uploader_key += 1  # new widget next time
                    st.session_state["uploaded_file"] = None
                    st.rerun()
                else:
                    st.error(response.json().get("message"))
        except Exception:
            log.exception("File upload failed")




# --- Query Interface ---

st.markdown("<h3>🧠🖼️ Multimodal Query Interface</h3>", unsafe_allow_html=True)

with st.form("query_form"):
    user_query = st.text_input("Enter your query")
    selected_doc = st.selectbox("Select document to query: ", st.session_state.available_sources,index=None)  # optionally get from API
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_query.strip():
        st.error("Please enter a question.")
        time.sleep(2)
        st.rerun()
      

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
        try:
            data = response.json()
        except Exception:
            st.error("Backend returned non-JSON response.")
            st.write("Raw response:", response.text)
            st.stop()

        status = data.get("status", "Failed to answer question")
    #st.write(response.json().get("result", "No result found") if response.status_code == 200 else response.json().get("status", "Failed to answer question (Internal_error)"))


st.markdown("<h3>🗑️ Manage Collection</h3>", unsafe_allow_html=True)

if st.session_state.available_sources:
    if st.button("Delete your document uploaded in DB"):
        with st.spinner("Deleting your collection..."):
                response = requests.delete(f"{API_URL}/delete_collection",params={"user_id": user_id},timeout=120)
                if response.status_code == 200:
                    st.success(response.json().get("message", "Collection deleted successfully."))
                    get_sources.clear()
                    st.session_state["sources_loaded"] = False
                    st.session_state.pop("selected_doc", False)
                    st.rerun()
                else:
                    result = response.json().get("message", "Failed to delete collection.")
                    st.error(result)

else:
    st.info("No collection in DB exists. Upload your documents to use RAG.")


st.markdown("<h3>🚪 Log out from your session</h3>", unsafe_allow_html=True)

if  st.button("Logout to wipe out your current session and session memory", type = "primary"):
    with st.spinner("Logging out..."):
        response = requests.post(f"{API_URL}/logout", headers = st.session_state.get("auth_headers"))
        if response.status_code == 200:
            st.write(response.json().get("message"))
            st.session_state["logging_out"] = True 
            st.session_state.pop("user", None)
            st.session_state.pop("jwt_token", None)
            st.session_state.pop("auth_headers", None)
            cookie_manager.delete("user", key="cookie_user_setter")
            cookie_manager.delete("jwt_token", key="cookie_jwt_setter")
            st.cache_data.clear()
            #st.session_state.pop("available_sources", None)
            st.rerun()
        else:
            st.session_state["logging_out"] = True
            st.session_state.clear()
            st.rerun()








   

  




   

        
        
        

  
        





