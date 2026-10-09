import os
import subprocess
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def get_vram_usage():
    """Queries rocm-smi via JSON output to pull precise VRAM analytics."""
    try:
        # Run rocm-smi with JSON output for robust, error-free parsing
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        data = json.loads(result.stdout)
        
        # Extract fields (rocm-smi keys usually look like card0, device0, etc.)
        device_key = list(data.keys())[0] 
        vram_used = int(data[device_key]["VRAM Total Memory Used (B)"])
        vram_total = int(data[device_key]["VRAM Total Memory Total (B)"])
        
        # Convert bytes to Gigabytes
        used_gb = vram_used / (1024 ** 3)
        total_gb = vram_total / (1024 ** 3)
        
        return used_gb, total_gb
    except Exception:
        # Fallback to a basic string parse if the JSON flag behaves oddly on your ROCm version
        try:
            result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
            lines = result.stdout.split("\n")
            # Parse lines searching for 'used' and 'total' keywords
            used = [line for line in lines if "used" in line.lower()][0].split()[-1]
            total = [line for line in lines if "total" in line.lower()][0].split()[-1]
            return float(used)/(1024**3), float(total)/(1024**3)
        except Exception:
            return 0.0, 8.0 # Safe default fallback for an 8GB RX 6600

# --- Build the Streamlit Sidebar Widget ---
st.sidebar.title("📊 AMD GPU Monitor")

# Get real-time stats
used_vram, total_vram = get_vram_usage()
vram_percentage = min(1.0, used_vram / total_vram)

# Display a clean metric block
st.sidebar.metric(
    label="VRAM Allocation", 
    value=f"{used_vram:.2f} GB / {total_vram:.2f} GB",
    delta=f"{(total_vram - used_vram):.2f} GB Free",
    delta_color="normal" if vram_percentage < 0.85 else "inverse"
)

# Render a native progress bar visualizer
st.sidebar.progress(vram_percentage)

# Optional: Add a manual refresh button to sync the stats
if st.sidebar.button("🔄 Refresh VRAM Stats"):
    st.rerun()

st.set_page_config(page_title="Local Document RAG Chatbot", layout="wide")
st.title("🤖 Local Document Chatbot (LCEL)")

# Initialize Ollama local model and embeddings
# Ensure you have run: ollama pull llama3.2
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# File uploader in Streamlit sidebar
uploaded_file = st.sidebar.file_uploader("Upload a .txt, .md,  or .pdf file", type=["txt", "md", "pdf"])

if uploaded_file:
    # Save file temporarily

    # make sure there is a tmp dir
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
        
    file_path = os.path.join("./tmp", uploaded_file.name)
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
