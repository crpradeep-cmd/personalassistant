import os
import shutil
import configparser
from operator import itemgetter

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def ingest_documents(doc_path, db_path, embedding_model):
    print(f"\n--- Ingesting: {doc_path} ---")
    loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader}
    documents = []
    for ext, loader_cls in loaders.items():
        loader = DirectoryLoader(doc_path, glob=f"**/*{ext}", loader_cls=loader_cls)
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext}: {e}")

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_chat_system(db, llm_model):
    llm = OllamaLLM(model=llm_model, temperature=0.5)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # --- LCEL PIPELINE ---
    
    # 1. Contextualize Question: Re-phrase the question based on history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question. Do NOT answer, just reformulate."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # This sub-chain produces a standalone string
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 2. Main Answer Chain
    qa_system_prompt = (
        "You are a professional assistant. Answer based ONLY on the context below.\n\n"
        "CONTEXT:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 3. Final Combined Chain (The "RAG Pipe")
    def get_contextualized_input(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_chain
        return itemgetter("input")

    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("input") | retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # 4. Chat Loop
    chat_history = []
    print(f"\n--- Modern LCEL Ready ({llm_model}) ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']: break

        # Run the LCEL pipe
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        print(f"\nAI: {response}")
        
        # Simple local state management
        chat_history.extend([("human", user_input), ("assistant", response)])

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

    run_chat_system(vector_db, settings['LLM_MODEL'])