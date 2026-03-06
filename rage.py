import os
import configparser
from operator import itemgetter 
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from paths import CONFIG_PATH, DOCUMENTS_DIR, CHROMA_DIR

def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def ingest_documents(doc_path, db_path, embedding_model):
    """
    Ingests PDF and DOCX documents from a folder into a Chroma DB.
    Optimized for Streamlit Admin usage.
    """
    if not os.path.exists(doc_path):
        return None

    loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader}
    documents = []
    
    for ext, loader_cls in loaders.items():
        # DirectoryLoader requires explicit glob for extensions
        loader = DirectoryLoader(
            doc_path, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            show_progress=True,
            use_multithreading=True
        )
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext} files in {doc_path}: {e}")

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Chroma.from_documents automatically saves (persists) to db_path
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    return vector_db

def format_docs(docs):
    """Helper to merge retrieved document contents."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_db, llm_model):
    llm = OllamaLLM(model=llm_model, temperature=0.5)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 1. Logic to handle Chat History (Contextualization)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 2. Main QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based ONLY on the provided context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 3. Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. The Parallel Chain (This is the "Brain")
    # It fetches the answer AND the raw documents at the same time.
    rag_chain = RunnableParallel({
        "answer": (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(
                    contextualize_chain.invoke(x) if x.get("chat_history") else x["input"]
                ))
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        ),
        "docs": itemgetter("input") | retriever
    })
    
    return rag_chain

# This block allows you to still test the file independently via Terminal
if __name__ == "__main__":
    config = load_config()
    settings = config['SETTINGS']
    folders = config['FOLDERS']
    
    options = list(folders.keys())
    print("\nSelect Context:")
    for i, opt in enumerate(options): print(f"{i+1}. {opt}")
    
    choice = int(input("Choice: ")) - 1
    selected_key = options[choice]
    
    doc_path = folders[selected_key]
    db_path = os.path.join(settings.get('CHROMA_BASE_PATH', '../chroma'), selected_key.lower())

    if not os.path.exists(db_path):
        vector_db = ingest_documents(doc_path, db_path, settings['EMBEDDING_MODEL'])
    else:
        print("Loading Existing DB...")
        emb = OllamaEmbeddings(model=settings['EMBEDDING_MODEL'])
        vector_db = Chroma(persist_directory=db_path, embedding_function=emb)

    # Simple terminal test loop
    chain = get_rag_chain(vector_db, settings['LLM_MODEL'])
    history = []
    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ['exit', 'quit']: break
        res = chain.invoke({"input": u_input, "chat_history": history})
        print(f"AI: {res}")
        history.extend([("human", u_input), ("assistant", res)])