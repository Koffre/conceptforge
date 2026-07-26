# src/main.py
"""
Streamlit interface for ConceptForge.

This module provides a chat interface that connects to the ConceptForge agent.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """Load a document using the RAG engine."""
    try:
        rag = get_rag_instance()
        rag.load_pdf_and_index(file_path)
        st.session_state.document_loaded = True
        st.session_state.current_document = os.path.basename(file_path)
        return True
    except Exception as e:
        st.error(f"❌ Error loading document: {e}")
        return False


def handle_chat(prompt: str):
    """Process a chat message and generate a response."""
    try:
        agent = get_agent_instance()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        response = agent.invoke(
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
        with st.spinner("📄 Processing document..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            if load_document(temp_path):
                st.success(f"✅ Document loaded: {uploaded_file.name}")
                st.session_state.document_name = uploaded_file.name
            else:
                st.error("❌ Failed to load document")
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
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
                response = handle_chat(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


