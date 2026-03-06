import streamlit as st
import configparser
import os
import requests
from paths import CONFIG_PATH, CHROMA_DIR
from langchain_ollama import OllamaEmbeddings

from rage import get_rag_chain, Chroma 

st.set_page_config(page_title="HOA Chat", layout="wide")

# Load Config
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

st.title("💬 My Personal Assistant")

# Sidebar for Context Selection
if 'FOLDERS' in config:
    available_contexts = list(config['FOLDERS'].keys())
    selected = st.sidebar.selectbox("Select Document Context", options=available_contexts)

    if selected:
        db_path = os.path.join(CHROMA_DIR, selected.lower())
        
        if os.path.exists(db_path):
            # 1. Initialize DB and Chain if not already in session
            if "rag_chain" not in st.session_state or st.session_state.get("current_ctx") != selected:
                with st.spinner("Loading AI Brain..."):
                    emb = OllamaEmbeddings(model=config['SETTINGS']['EMBEDDING_MODEL'])
                    vector_db = Chroma(persist_directory=db_path, embedding_function=emb)
                    
                    # Create the chain using the function in rage.py
                    st.session_state.rag_chain = get_rag_chain(vector_db, config['SETTINGS']['LLM_MODEL'])
                    st.session_state.current_ctx = selected
                    st.session_state.messages = [] # Reset chat history for new context

            # 2. Display Chat History
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            # 3. Chat Input
            if prompt := st.chat_input("Ask me anything about " + selected):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 4. Generate Response
                with st.chat_message("assistant"):
                    # Format history for the LCEL chain (list of tuples)
                    formatted_history = [
                        (m["role"], m["content"]) for m in st.session_state.messages[:-1]
                    ]
                    
                    # Invoke the chain from rage.py
                    response = st.session_state.rag_chain.invoke({
                        "input": prompt, 
                        "chat_history": formatted_history
                    })
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning(f"No database found for {selected}. Please go to the Admin Panel to index documents first.")
else:
    st.error("No contexts found in config.ini. Please use the Admin Panel to create one.")
    
    
if prompt := st.chat_input("Ask me anything about " + selected, key=f"chat_input_{selected}"):
    # Add user message to UI state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # --- 4. Generate Response ---
    with st.chat_message("assistant"):
        # ADD THIS LINE: Convert history into the format LangChain understands
        formatted_history = [
            (m["role"], m["content"]) for m in st.session_state.messages[:-1]
        ]
        
        # Now formatted_history exists and can be used here:
        full_result = st.session_state.rag_chain.invoke({
            "input": prompt, 
            "chat_history": formatted_history
        })
        
        answer = full_result["answer"]
        st.markdown(answer)
        
        # Display sources if available
        if "docs" in full_result:
            with st.expander("📚 View Sources"):
                for doc in full_result["docs"]:
                    st.write(f"- {os.path.basename(doc.metadata.get('source', 'Unknown'))}")

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})