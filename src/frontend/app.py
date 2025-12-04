import streamlit as st
import requests
import json
import os
import jwt,time
from extra_streamlit_components import CookieManager
from logger_config import log


st.set_page_config(page_title="RAG App Login", page_icon="🔐")
st.markdown("<h2>⚡🧠 Advanced Agentic RAG + ChatGPT</h2>", unsafe_allow_html=True)

# -------------------- BACKEND BASE URL --------------------
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
JWT_SECRET = st.secrets["JWT_SECRET"]
JWT_ALGO = "HS256"
cookie_manager = CookieManager()

# Initialize session state
st.session_state.setdefault("user", None)
st.session_state.setdefault("jwt_token", None)
st.session_state.setdefault("auth_headers", None)
st.session_state.setdefault("sources_loaded", False)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("uploaded_file", [])
st.session_state.setdefault("selected_doc", [])


user_cookie = cookie_manager.get("user") or None
token_cookie = cookie_manager.get("jwt_token") or None

# Validate the token and load user information
if user_cookie and token_cookie:
    try:
        # Decode the JWT token
        decoded = jwt.decode(token_cookie, JWT_SECRET, algorithms=[JWT_ALGO])
        st.session_state["user"] = user_cookie
        st.session_state["jwt_token"] = token_cookie
    except (jwt.InvalidTokenError, json.JSONDecodeError):
        # If the token is invalid, clear the session state
        st.session_state.clear()

# Handle login via OAuth callback
if not st.session_state["user"] and not st.session_state["jwt_token"]:
    token_list = st.query_params.get("token")
    if token_list:
        try:
            decoded = jwt.decode(token_list, JWT_SECRET, algorithms=[JWT_ALGO])
            user_info = decoded["user"]
            st.session_state["user"] = user_info
            st.session_state["jwt_token"] = token_list

            # Store cookies for the user
            cookie_manager.set("user", json.dumps(user_info), key="cookie_user_setter")
            cookie_manager.set("jwt_token", token_list, key="cookie_jwt_setter")

            # Clear query parameters to prevent processing again
            st.query_params.clear()
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            st.error("Invalid or expired login token.")
            st.stop()

# Redirect to login if not authenticated
if not st.session_state["user"]:
    st.subheader("Please log in to continue")
    login_url = f"{API_URL}/login"
    st.markdown(f'<a href="{login_url}" target="_self">👉 Log in with Google</a>', unsafe_allow_html=True)
    st.stop()

# If a user is logged in, display their information
user = st.session_state["user"]
jwt_token = st.session_state["jwt_token"]
st.success(f"Welcome {user['name']} 👋 ({user['email']})")
st.session_state.auth_headers = {"Authorization": f"Bearer {jwt_token}"}
user_id = user["sub"]

# --- Available Sources ---
def get_sources():
    if "auth_headers" not in st.session_state or st.session_state["auth_headers"] is None:
        st.warning("Please log in first.")
        return []
    
    response = requests.get(f"{API_URL}/available_sources", headers=st.session_state["auth_headers"])
    if response.status_code == 200:
        return response.json().get("sources", [])
    else:
        #log.info("No available resources found")
        return response.json().get("sources", [])

if not st.session_state["sources_loaded"]:
    st.session_state["available_sources"] = get_sources()
    st.session_state["sources_loaded"] = True

# --- File Upload Section ---
st.markdown("<h3>📁 Upload File to Build Your RAG System</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose file",
    type=["xlsx", "docx", "pptx", "csv", "txt", "pdf"],
    key=f"upload_{st.session_state['uploader_key']}")

if uploaded_file and st.button("Upload File"):
    try:
        with st.spinner("Processing file..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload_file", files=files, headers=st.session_state["auth_headers"])
            if response.status_code == 200:
                st.success(response.json().get("message"))
                st.session_state["available_sources"] = get_sources()
                st.session_state.uploader_key += 1 
                st.session_state["uploaded_file"] = None
                st.rerun()  # Refresh the list
            else:
                st.error(response.json().get("message"))
    except Exception as e:
        st.error("File upload failed.")
        st.exception(e)

# --- Query Interface ---
st.markdown("<h3>🧠🖼️ Multimodal Query Interface</h3>", unsafe_allow_html=True)
with st.form("query_form"):
    user_query = st.text_input("Enter your query")
    selected_doc = st.selectbox("Select document to query: ", st.session_state["available_sources"], index=None)
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Submit")

if submitted:
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        data = {
            "query": user_query,
            "selected_doc": selected_doc
        }

        
        if uploaded_image:
            files = {'image': (uploaded_image.name, uploaded_image.getvalue(),uploaded_image.type)}
        else:
            files = {}

        data = {
                "query":user_query,
                "selected_doc" : selected_doc 
            }
        
        with st.spinner("Processing query..."):
            response = requests.post(f"{API_URL}/query", data=data, headers=st.session_state["auth_headers"])
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



st.markdown("<h3>🗑️ Manage Collection</h3>", unsafe_allow_html=True)

if st.session_state.available_sources:
    if st.button("Delete your document uploaded in DB"):
        with st.spinner("Deleting your collection..."):
            response = requests.delete(f"{API_URL}/delete_collection",params={"user_id": user_id},timeout=120)
            if response.status_code == 200:
                st.success(response.json().get("message", "Collection deleted successfully."))
                st.session_state["available_sources"] = []
                st.session_state["sources_loaded"] = False
                st.session_state.pop("selected_doc", False)
                st.rerun()
            else:
                result = response.json().get("message", "Failed to delete collection.")
                st.error(result)

else:
    st.info("No collection in DB exists. Upload your documents to use RAG.")


# Debugging: Check current cookie state before deletion
# --- Logout Section ---
st.markdown("<h3>🚪 Log out from your session</h3>", unsafe_allow_html=True)
if st.button("Logout", key="logout_button"):
    with st.spinner("Logging out..."):
        try:
            response = requests.post(f"{API_URL}/logout", headers={"Authorization": f"Bearer {jwt_token}"})
    
            
            # Debugging: Show backend response status (can be removed later
            log.info("Pre deletion")
            log.info(f"{cookie_manager.get('user')}")
            log.info(f"{cookie_manager.get('jwt_token')}") 

            cookie_manager.delete("user", key="cookie_user_setter")
            cookie_manager.delete("jwt_token", key="cookie_jwt_setter")


            st.session_state.clear() 
            
            log.info("Post del")
            log.info(f"{cookie_manager.get('user')}")
            log.info(f"{cookie_manager.get('jwt_token')}")
         

             # Hard redirect to LOGIN page (NOT rerun!)
            login_url = f"{API_URL}/login"

            st.write(f"""
                <meta http-equiv="refresh" content="0; url={login_url}" />
            """, unsafe_allow_html=True)

            st.stop()
            
                    
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Please check your network or try again later.")
            # Still clear frontend state and redirect to login, as the user can't interact meaningfully.
            cookie_manager.delete("user", key="cookie_user_setter")
            cookie_manager.delete("jwt_token", key="cookie_jwt_setter")
            st.session_state.clear() 
            log.info("Post del")
            log.info(f"{cookie_manager.get('user')}")
            log.info(f"{cookie_manager.get('jwt_token')}")
         
             # Hard redirect to LOGIN page (NOT rerun!)
            login_url = f"{API_URL}/login"

            st.write(f"""
                <meta http-equiv="refresh" content="0; url={login_url}" />
            """, unsafe_allow_html=True)

            st.stop()

        except Exception as e:
            # Catch any other unexpected exceptions
            log.exception(f"An unexpected error occurred during logout: {e}")
            st.error("An unexpected error occurred during logout. Please try again.")
            cookie_manager.delete("user", key="cookie_user_setter")
            cookie_manager.delete("jwt_token", key="cookie_jwt_setter")
            st.session_state.clear() 
            log.info("Post del")
            log.info(f"{cookie_manager.get('user')}")
            log.info(f"{cookie_manager.get('jwt_token')}")
         

             # Hard redirect to LOGIN page (NOT rerun!)
            login_url = f"{API_URL}/login"

            st.write(f"""
                <meta http-equiv="refresh" content="0; url={login_url}" />
            """, unsafe_allow_html=True)

            st.stop()
         
     





   

  




   

        
        
        

  
        





