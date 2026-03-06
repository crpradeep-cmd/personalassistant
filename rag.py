import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# --- Configuration ---
DOCUMENTS_PATH = "../vbn"
CHROMA_DB_PATH = "../chromadb/vbn" 
LLM_MODEL = "llama3:8b" 
EMBEDDING_MODEL = "nomic-embed-text"

# --- Ingestion Pipeline ---
def ingest_documents():
    print(f"Loading documents from {DOCUMENTS_PATH}...")
    
    # Using PyPDFLoader to avoid Poppler/System dependency issues
    loader = DirectoryLoader(
        DOCUMENTS_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

    if not documents:
        print("No documents found. Please check the DOCUMENTS_PATH.")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Create Ollama Embedding Instance
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Store embeddings in ChromaDB
    print("Creating Vector Store (this may take a few minutes)...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_PATH
    )
    print(f"Vector Store created at {CHROMA_DB_PATH}.")
    return db

# --- Main Query Function ---
def run_query_system(db):
    if db is None:
        print("Cannot run query: Vector store is not available.")
        return

    # Updated to OllamaLLM for LangChain 0.3
    llm = OllamaLLM(model=LLM_MODEL)

    RAG_PROMPT_TEMPLATE = """
    You are an AI assistant that answers questions based ONLY on the provided context.
    If the context does not contain the answer, you must state: "I cannot find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    """

    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)}
    )
    
    print("\n--- RAG System Ready ---")
    
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == 'exit':
            break
        
        response = qa_chain.invoke({"query": question})
        
        print("\n**AI Answer**:")
        print(response['result'])
        
        if response.get('source_documents'):
            sources = {doc.metadata.get('source') for doc in response['source_documents']}
            print("\n**Sources Used:**")
            for source in sources:
                print(f"- {source}")
        
        print("-" * 50)

if __name__ == "__main__":
    # --- FORCE RE-INDEX LOGIC ---
    # Deletes existing DB to ensure a fresh index of your folder
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Cleaning up existing database at {CHROMA_DB_PATH}...")
        shutil.rmtree(CHROMA_DB_PATH)

    # Run ingestion
    vector_db = ingest_documents()
    
    # Start the query interface
    run_query_system(vector_db)