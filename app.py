import os
import re
import subprocess
import json
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# --- Page Configuration ---
st.set_page_config(page_title="Local Chatbot", layout="wide")

# Initialize Session State Variables at Top Level
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "last_prompt_tokens" not in st.session_state:
    st.session_state.last_prompt_tokens = 0
if "last_completion_tokens" not in st.session_state:
    st.session_state.last_completion_tokens = 0

def render_message_with_code_downloads(content, message_idx=0):
  """Parses a message for markdown code blocks and adds download buttons."""
  # Split the text by markdown code blocks, keeping the blocks in the list
  parts = re.split(r"(```[\s\S]*?```)", content)

  for i, part in enumerate(parts):
    if part.startswith("```") and part.endswith("```"):
      # Extract language and code lines
      lines = part.strip().split("\n")
      first_line = lines[0][3:].strip()
      language = first_line if first_line else "python"
      code_content = "\n".join(lines[1:-1])

      # Render the code block
      st.code(code_content, language=language)

      # Map language to standard file extensions
      ext_map = {
          "python": "py",
          "py": "py",
          "javascript": "js",
          "js": "js",
          "html": "html",
          "css": "css",
          "json": "json",
          "sh": "sh",
          "bash": "sh",
          "sql": "sql",
      }
      ext = ext_map.get(language.lower(), "txt")

      # Unique key required for Streamlit widgets inside loops
      unique_key = f"dl_msg_{message_idx}_code_{i}"

      st.download_button(
          label=f"📥 Download {language.capitalize()} Script",
          data=code_content,
          file_name=f"generated_script.{ext}",
          mime="text/plain",
          key=unique_key,
      )
    else:
      if part.strip():
        st.write(part)

def get_vram_usage():
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
        return vram_used / (1024 ** 3), vram_total / (1024 ** 3)
    except Exception:
        return 0.0, 8.0

def get_installed_ollama_models():
    models = []
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        if res.status_code == 200:
            data = res.json()
            models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    chat_models = [m for m in models if "embed" not in m.lower()]
    return chat_models if chat_models else ["llama3.2"]

# --- Sidebar UI ---
st.sidebar.title("⚙️ LLM & System Settings")

available_models = get_installed_ollama_models()
selected_model = st.sidebar.selectbox(
    "🤖 Select Installed LLM",
    options=available_models,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.title("📊 AMD GPU Monitor")

used_vram, total_vram = get_vram_usage()
vram_percentage = min(1.0, used_vram / total_vram)
st.sidebar.metric(
    label="VRAM Allocation", 
    value=f"{used_vram:.2f} GB / {total_vram:.2f} GB",
    delta=f"{(total_vram - used_vram):.2f} GB Free"
)
st.sidebar.progress(vram_percentage)

st.sidebar.markdown("---")
st.sidebar.title("🔢 Token Analytics")
st.sidebar.metric("Last Prompt Tokens", st.session_state.last_prompt_tokens)
st.sidebar.metric("Last Completion Tokens", st.session_state.last_completion_tokens)
st.sidebar.metric("Cumulative Tokens Used", st.session_state.total_tokens)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload a .txt, .md, .py, or .pdf file", type=["txt", "py", "md", "pdf"])

if uploaded_file:
    default_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, just say that you don't know.\n\n"
        "Context:\n{context}"
    )
else:
    default_system_prompt = "You are a helpful, smart, and concise AI assistant."

system_prompt = st.sidebar.text_area("💬 System Prompt", value=default_system_prompt, height=160)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.last_prompt_tokens = 0
    st.session_state.last_completion_tokens = 0
    st.rerun()

# --- Model & Document Processing ---
chat_llm = ChatOllama(model=selected_model, num_ctx=8192)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

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

    if uploaded_file.name.endswith(".py"):
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    if os.path.exists(file_path):
        os.remove(file_path)

# --- Main Interface ---
title_suffix = f" ({selected_model} - RAG)" if uploaded_file else f" ({selected_model} - Open Chat)"
st.title(f"🤖 Local Chatbot{title_suffix}")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
          render_message_with_code_downloads(message["content"], message_idx=idx)
        else:
          st.write(message["content"])

# --- Chat Handling ---
user_query = st.chat_input("Ask a question or start chatting:")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        # 1. Build chat history message objects for LangChain
        chat_history_msgs = []
        # Exclude the current user query since it goes into the 'input' variable
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history_msgs.append(HumanMessage(content=msg["content"]))
            else:
                chat_history_msgs.append(AIMessage(content=msg["content"]))

        # 2. Define standard prompt layout with MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        if retriever:
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "chat_history": lambda x: x["chat_history"],
                    "input": lambda x: x["input"]
                }
                | prompt
                | chat_llm
            )
            ai_message = rag_chain.invoke({
                "chat_history": chat_history_msgs,
                "input": user_query
            })
        else:
            chat_chain = prompt | chat_llm
            ai_message = chat_chain.invoke({
                "chat_history": chat_history_msgs,
                "input": user_query
            })

        response_text = ai_message.content
        render_message_with_code_downloads(
            response_text, message_idx=len(st.session_state.messages)
        )

        # Robust extraction supporting standard LangChain and Ollama-native response metadata
        response_meta = getattr(ai_message, "response_metadata", {})
        usage_metadata = getattr(ai_message, "usage_metadata", {}) or response_meta.get("token_usage", {})

        p_tokens = (
            usage_metadata.get("prompt_tokens") 
            or response_meta.get("prompt_eval_count") 
            or 0
        )
        c_tokens = (
            usage_metadata.get("completion_tokens") 
            or response_meta.get("eval_count") 
            or 0
        )
        
        st.session_state.last_prompt_tokens = p_tokens
        st.session_state.last_completion_tokens = c_tokens
        st.session_state.total_tokens += (p_tokens + c_tokens)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()
