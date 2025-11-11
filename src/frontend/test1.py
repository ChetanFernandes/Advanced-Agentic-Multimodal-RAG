import streamlit as st
import uuid
from streamlit_js_eval import streamlit_js_eval

# Try to get from localStorage
ls_result = streamlit_js_eval(js_expressions="localStorage.getItem('user_id')", key="ls_uid")

# On first run, ls_result is None until component runs JS and returns!
if ls_result is None:
    st.info("Initializing your session, please wait…")
    st.stop()

local_uid = ls_result["result"]

query_uid = st.query_params.get("uid", [None])[0]

if "user_id" not in st.session_state:
    if query_uid:
        st.session_state.user_id = query_uid
    elif local_uid:
        st.session_state.user_id = local_uid
        st.query_params["uid"] = local_uid
        st.rerun()
    else:
        new_id = uuid.uuid4().hex[:8]
        st.session_state.user_id = new_id
        st.query_params["uid"] = new_id
        streamlit_js_eval(js_expressions=f"localStorage.setItem('user_id', '{new_id}')", key="ls_set")
        st.rerun()

# Always use session_state
user_id = st.session_state.user_id

st.success(f"✅ Your persistent user ID: {user_id}")
# --- UI HEADER --
st.title("Advanced_RAG + Chat_GPT")
st.header("Upload File to Build a RAG System")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["xlsx", "docx", "pptx", "csv", "txt", "pdf"],
    key=f"upload_{st.session_state['uploader_key']}"
)