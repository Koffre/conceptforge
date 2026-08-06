# src/main.py
"""
Streamlit interface for ConceptForge.

This module provides a chat interface that connects to the ConceptForge agent.
"""

import os
import sys
import tempfile
import asyncio
import nest_asyncio
from pathlib import Path

import streamlit as st
from contextlib import contextmanager

# Compatibility shims: implement missing Streamlit-like helpers if absent
if not hasattr(st, "divider"):
    def divider():
        st.markdown("---")
    st.divider = divider

if not hasattr(st, "chat_input"):
    def chat_input(prompt: str):
        # Fallback to a simple text_input for environments without chat_input
        return st.text_input(prompt, key="_chat_input")
    st.chat_input = chat_input

if not hasattr(st, "chat_message"):
    @contextmanager
    def chat_message(role: str):
        # Provide a simple container context for chat messages
        container = st.container()
        with container:
            yield
    st.chat_message = chat_message

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

nest_asyncio.apply()

from src.agent import get_agent
from src.rag_engine import RAGEngine
from langchain.messages import HumanMessage

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ConceptForge",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .tool-header {
        font-weight: 600;
        color: #1f77b4;
        margin-top: 0.5rem;
    }
    .stChatMessage {
        padding: 1rem;
    }
    /* Make sidebar narrower and cleaner */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        padding: 1rem 0.5rem;
    }
    
    /* Hide the default sidebar collapse button if you want */
    .stSidebarCollapseButton {
        display: none !important;
    }
    
    /* Better spacing for sidebar content */
    .stSidebar .stMarkdown {
        padding: 0 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if "agent" not in st.session_state:
    st.session_state.agent = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "current_document" not in st.session_state:
    st.session_state.current_document = None


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_agent_instance():
    """Get or create the agent instance."""
    if st.session_state.agent is None:
        with st.spinner("🔄 Initializing ConceptForge agent..."):
            st.session_state.agent = get_agent()
    return st.session_state.agent


def get_rag_instance():
    """Get or create the RAG engine instance."""
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine()
    return st.session_state.rag_engine


def load_document(file_path: str) -> bool:
    """Load a document using the RAG engine and share it with MCP."""
    try:
        rag = get_rag_instance()
        rag.load_pdf_and_index(file_path)
        st.session_state.document_loaded = True
        st.session_state.current_document = os.path.basename(file_path)
        from src.mcp_server import set_rag_engine
        set_rag_engine(rag)
        return True
    except Exception as e:
        st.error(f"❌ Error loading document: {e}")
        return False


async def handle_chat(prompt: str):
    """Process a chat message and generate a response."""
    try:
        agent = get_agent_instance()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )
        
        # Extract the response
        last_message = response['messages'][-1]
        if isinstance(last_message.content, list):
            answer = last_message.content[0].get('text', str(last_message.content))
        else:
            answer = last_message.content
        
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"


def clear_conversation():
    """Clear the conversation history."""
    st.session_state.messages = []
    st.session_state.thread_id = str(int(st.session_state.thread_id) + 1)


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.markdown("## ⚙️ ConceptForge")
    st.markdown("*Intelligent Research Assistant*")
    st.divider()
    
    # Document Upload
    st.markdown("### 📄 Document Management")
    
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a PDF to analyze and query",
    key="file_uploader"
)

if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if load_document(file_path):
        st.success(f"✅ Document loaded: {uploaded_file.name}")
        st.session_state.document_loaded = True
        st.session_state.current_document = uploaded_file.name
        
        with st.spinner("🔄 Indexando en el servidor MCP..."):
            agent = get_agent_instance()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            load_message = f"Please load the document from the path: {file_path} using the load_document tool. Do not ask for confirmation, just load it."
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    agent.ainvoke(
                        {"messages": [HumanMessage(content=load_message)]},
                        config=config
                    )
                )
                last_msg = response['messages'][-1]
                if isinstance(last_msg.content, list):
                    answer = last_msg.content[0].get('text', '')
                else:
                    answer = last_msg.content
                st.info(f"📡 MCP Server: {answer}")
            except Exception as e:
                st.error(f"❌ Error al cargar en MCP: {e}")
            finally:
                loop.close()
                asyncio.set_event_loop(None)
    else:
        st.error("❌ Failed to load document")
    
    # Document status
    if st.session_state.document_loaded:
        st.markdown("---")
        st.markdown("### 📊 Document Status")
        st.markdown(f"**File:** {st.session_state.current_document}")
        
        rag = get_rag_instance()
        if rag.vector_store:
            st.success("✅ Vector store ready")
            st.info(f"📄 Chunks: {len(rag.documents) if rag.documents else 0}")
    
    st.divider()
    # --- Help Menu (NEW) ---
    st.markdown("### 📚 Commands Help")
    with st.expander("💬 Available commands"):
        st.markdown("""
        **Ask about your document:**
        
        - `resume el documento` — Get a detailed summary
        - `genera un mapa conceptual` — Generate a concept map
        - `busca información sobre [tema]` — Search for specific info
        - `¿De qué trata este documento?` — General overview
        
        **In English:**
        - `summarize the document`
        - `generate a concept map`
        - `search for [topic]`
        """)
    st.divider()

    # Controls
    st.markdown("### 🎛️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_conversation()
        st.rerun()
    
    st.divider()
    
    # Information
    with st.expander("ℹ️ About ConceptForge"):
        st.markdown("""
        **ConceptForge** is an intelligent research assistant that uses:
        
        - **RAG** for semantic search
        - **MCP** for tool integration
        - **LangChain** for agent orchestration
        - **Gemini** for LLM capabilities
        
        **Features:**
        - Semantic document search
        - Automatic summarization
        - Concept map generation
        - Multi-turn conversation
        """)


# ============================================
# MAIN CHAT INTERFACE
# ============================================

# Header
st.markdown('<p class="main-header">⚙️ ConceptForge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a document and start exploring its content</p>', unsafe_allow_html=True)

with st.expander("💡 What can I do with ConceptForge?", expanded=True):
    st.markdown("""
    **Suggested commands:**
    
    - 📝 **Summarize document**
    - 🧠 **Concept map generation**
    - 🔍 **Topic search**
    """)

# Status indicator
if st.session_state.document_loaded:
    st.info(f"📄 Document loaded: **{st.session_state.current_document}**")
else:
    st.warning("📄 No document loaded. Upload a PDF to start analyzing.")

# Chat History
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if a document is loaded
    if not st.session_state.document_loaded:
        with st.chat_message("assistant"):
            error_msg = "⚠️ Please upload a document first before asking questions."
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    agent = get_agent_instance()
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(
                            agent.ainvoke(
                                {"messages": [HumanMessage(content=prompt)]},
                                config=config
                            )
                        )
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
                    
                    # Extraer la respuesta
                    last_message = response['messages'][-1]
                    if isinstance(last_message.content, list):
                        answer = last_message.content[0].get('text', str(last_message.content))
                    else:
                        answer = last_message.content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

