import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import os
import configparser
import shutil
import pandas as pd
from paths import CONFIG_PATH, DOCUMENTS_DIR, CHROMA_DIR
from rage import ingest_documents 


# --- Page Setup ---
st.set_page_config(page_title="Admin Panel", layout="wide")
st.title("🛡️ System Administration")
st.markdown("Manage document contexts, upload files, and trigger database indexing.")

# Load Config
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# Ensure sections exist
if 'FOLDERS' not in config: config.add_section('FOLDERS')
if 'SETTINGS' not in config: config.add_section('SETTINGS')

# --- 1. CONTEXT CREATION ---
with st.sidebar:
    st.header("✨ New Context")
    new_ctx = st.text_input("Enter Name (e.g., VBT)").upper().strip()
    if st.button("Add Context"):
        if new_ctx and new_ctx not in config['FOLDERS']:
            new_folder_path = os.path.join(DOCUMENTS_DIR, new_ctx.lower())
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Save to config
            config.set('FOLDERS', new_ctx, new_folder_path)
            with open(CONFIG_PATH, 'w') as f: config.write(f)
            st.success(f"Context '{new_ctx}' created!")
            st.rerun()
        else:
            st.error("Context already exists or name is empty.")

# --- 2. FILE MANAGEMENT ---
st.header("📂 Document Management")
available_contexts = list(config['FOLDERS'].keys())

if not available_contexts:
    st.info("No contexts available. Create one in the sidebar.")
else:
    selected_ctx = st.selectbox("Select Context to Manage", options=available_contexts)
    target_doc_path = config['FOLDERS'][selected_ctx]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            f"Add to {selected_ctx}", 
            accept_multiple_files=True, 
            type=['pdf', 'docx']
        )
        if st.button("Upload & Save"):
            if uploaded_files:
                for f in uploaded_files:
                    with open(os.path.join(target_doc_path, f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                st.success(f"Saved {len(uploaded_files)} files.")
                st.rerun()

    with col2:
        st.subheader("Existing Files")
        files_in_folder = os.listdir(target_doc_path)
        if files_in_folder:
            # Create a table with delete buttons
            for file_name in files_in_folder:
                c1, c2 = st.columns([3, 1])
                c1.text(file_name)
                if c2.button("🗑️", key=f"del_{file_name}"):
                    os.remove(os.path.join(target_doc_path, file_name))
                    st.rerun()
        else:
            st.write("Folder is empty.")

# --- 3. INDEXING CONTROL ---
st.divider()
st.header("🧠 Database Controls")

ctx_to_index = st.selectbox("Select Context to Index", options=available_contexts, key="index_sel")
db_status_path = os.path.join(CHROMA_DIR, ctx_to_index.lower())

if os.path.exists(db_status_path):
    st.warning(f"Database for {ctx_to_index} already exists at: `{db_status_path}`")
else:
    st.info(f"No database found for {ctx_to_index}. Retrieval will not work until indexed.")

if st.button("🚀 Start Indexing (Re-index)", type="primary"):
    # Delete old DB if it exists to ensure fresh start
    if os.path.exists(db_status_path):
        shutil.rmtree(db_status_path)
    
    with st.status(f"Indexing {ctx_to_index}...", expanded=True) as status:
        st.write("Loading files from disk...")
        # Call the ingest logic from your rage_logic file
        ingest_documents(
            config['FOLDERS'][ctx_to_index], 
            db_status_path, 
            config['SETTINGS']['EMBEDDING_MODEL']
        )
        status.update(label="Indexing Complete!", state="complete", expanded=False)
    st.balloons()