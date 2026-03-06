This README summarizes the setup for your **My Personal Assistant**, a RAG (Retrieval-Augmented Generation) system built with LangChain, Ollama, and Streamlit.

---

# 🏠 Personal Assistant

A professional document-chat system designed to index and query local documents (PDFs and DOCX) using local LLMs.

## 📋 Prerequisites

1. **Ollama**: Install [Ollama](https://ollama.com/) and download your models.
* Open terminal and run: `ollama pull llama3` (or your preferred model).
* Run: `ollama pull mxbai-embed-large` (for embeddings).


2. **Conda**: Ensure Anaconda or Miniconda is installed.

---

## 🛠️ 1. Installation & Environment Setup

Open your terminal (or Anaconda Prompt) and run the following commands to create a clean environment:

```bash
# Create the environment
conda create -n aip python=3.11 -y

# Activate the environment
conda activate aip

# Install Streamlit and required libraries
pip install streamlit langchain langchain-community langchain-ollama langchain-chroma pypdf docx2txt requests

```

---

## 📂 2. Project Structure

Ensure your project folder (`project`) is organized as follows:

```text
project/
├── .streamlit/
│   └── secrets.toml      # Admin password
├── pages/
│   └── 1_Admin_Panel.py  # Document management
├── Home.py               # Main Chat UI
├── rage.py               # AI Logic & LangChain Pipeline
├── paths.py              # Centralized path management
├── auth.py               # Password protection logic
├── config.ini            # App configuration
├── documents/            # Raw PDF/Docx files (created automatically)
└── chroma_db/            # Vector database files (created automatically)

```

---

## 🔑 3. Configuration & Secrets

### `.streamlit/secrets.toml`

Create this file to secure your Admin Panel:

```toml
[auth]
ADMIN_PASSWORD = "your_secure_password"

```

### `config.ini`

Define your models and document folders:

```ini
[SETTINGS]
LLM_MODEL = llama3
EMBEDDING_MODEL = mxbai-embed-large
CHROMA_BASE_PATH = chroma_db

[FOLDERS]
VBN = documents/vbn
VBS = documents/vbs

```

---

## 🚀 4. How to Run

1. **Start Ollama**: Ensure the Ollama application is running in your taskbar.
2. **Navigate to Project**:
```bash
cd D:\Projects\AI\project

```


3. **Launch Streamlit**:
```bash
streamlit run Home.py

```



---

## 📖 5. Using the App

1. **Admin Panel**: Go to the Admin Panel in the sidebar. Enter your password. Use this page to upload documents and click **"Start Indexing"**. This converts your documents into a searchable "Vector Brain."
2. **Chat (Home)**: Select the document context (e.g., VBN) from the sidebar. Ask questions about your documents.
3. **Sources**: The AI will provide answers based **only** on your documents. Click **"View Sources"** under the answer to see which files were used.

---

## 🛠️ Troubleshooting

* **Duplicate Element Error**: If the chat input crashes, ensure you are only calling `st.chat_input` once per page and that it has a unique `key`.
* **Ollama Offline**: If the app says Ollama is offline, check that your local Ollama server is running at `http://localhost:11434`.

* **ModuleNotFoundError**: Ensure you are running the command from the root folder (`project`) and not from inside `src` or `pages`.
