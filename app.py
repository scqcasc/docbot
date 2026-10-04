import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

st.set_page_config(page_title="Local Document RAG Chatbot", layout="wide")
st.title("🤖 Local Document Chatbot (LCEL)")

# Initialize Ollama local model and embeddings
# Ensure you have run: ollama pull llama3.2
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# File uploader in Streamlit sidebar
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    # Save file temporarily
    file_path = os.path.join(".", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Create local vector store index
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Setup RAG chain using Modern LCEL
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, just say that you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construct the LCEL chain pipeline
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat interface
    user_query = st.chat_input("Ask something about your document:")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            # Invoke the pipeline directly
            response_text = rag_chain.invoke(user_query)
            st.write(response_text)
            
    # Clean up temporary file on completion
    if os.path.exists(file_path):
        os.remove(file_path)
else:
    st.info("Please upload a document on the sidebar to get started.")
