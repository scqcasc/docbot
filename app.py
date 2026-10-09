import os
import subprocess
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Local Chatbot", layout="wide")

def get_vram_usage():
    """Queries rocm-smi via JSON output to pull precise VRAM analytics."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        data = json.loads(result.stdout)
        
        device_key = list(data.keys())[0] 
        vram_used = int(data[device_key]["VRAM Total Memory Used (B)"])
        vram_total = int(data[device_key]["VRAM Total Memory Total (B)"])
        
        used_gb = vram_used / (1024 ** 3)
        total_gb = vram_total / (1024 ** 3)
        
        return used_gb, total_gb
    except Exception:
        try:
            result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
            lines = result.stdout.split("\n")
            used = [line for line in lines if "used" in line.lower()][0].split()[-1]
            total = [line for line in lines if "total" in line.lower()][0].split()[-1]
            return float(used)/(1024**3), float(total)/(1024**3)
        except Exception:
            return 0.0, 8.0 # Safe default fallback

# --- Build the Streamlit Sidebar Widget ---
st.sidebar.title("📊 AMD GPU Monitor")

used_vram, total_vram = get_vram_usage()
vram_percentage = min(1.0, used_vram / total_vram)

st.sidebar.metric(
    label="VRAM Allocation", 
    value=f"{used_vram:.2f} GB / {total_vram:.2f} GB",
    delta=f"{(total_vram - used_vram):.2f} GB Free",
    delta_color="normal" if vram_percentage < 0.85 else "inverse"
)

st.sidebar.progress(vram_percentage)

if st.sidebar.button("🔄 Refresh VRAM Stats"):
    st.rerun()

st.sidebar.markdown("---")

# File uploader in Streamlit sidebar
uploaded_file = st.sidebar.file_uploader("Upload a .txt, .md, or .pdf file (Optional)", type=["txt", "md", "pdf"])

# Define default prompts depending on mode
if uploaded_file:
    default_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, just say that you don't know.\n\n"
        "Context:\n{context}"
    )
    help_text = "Modify how the model handles retrieved documents. Keep the {context} placeholder intact."
else:
    default_system_prompt = "You are a helpful, smart, and concise AI assistant."
    help_text = "Modify how the model behaves during open-ended chat."

# System prompt control widget
system_prompt = st.sidebar.text_area(
    label="⚙️ System Prompt",
    value=default_system_prompt,
    height=180,
    help=help_text
)

# Chat history management
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# --- Main Interface ---
title_suffix = " (Document RAG)" if uploaded_file else " (Open Chat)"
st.title(f"🤖 Local Chatbot{title_suffix}")

# Display chat message history on rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialize Ollama local model and embeddings
llm = ChatOllama(model="llama3.2")

# --- RAG Branch: Process Document if Uploaded ---
retriever = None
if uploaded_file:
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
        
    file_path = os.path.join("./tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    if os.path.exists(file_path):
        os.remove(file_path)

# --- Chat Handling ---
user_query = st.chat_input("Ask a question or start chatting:")

if user_query:
    # Append user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        if retriever:
            # RAG Mode Chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            response_text = rag_chain.invoke(user_query)
        else:
            # Direct Open-Ended Chat Mode (Includes full conversation history)
            history_messages = [("system", system_prompt)]
            for msg in st.session_state.messages:
                role = "human" if msg["role"] == "user" else "ai"
                history_messages.append((role, msg["content"]))

            prompt = ChatPromptTemplate.from_messages(history_messages)
            chat_chain = prompt | llm | StrOutputParser()
            response_text = chat_chain.invoke({})

        st.write(response_text)

    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
