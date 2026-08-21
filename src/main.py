
"""
ConceptForge - Streamlit Application

Main user interface for the ConceptForge research assistant.
"""

import os
import sys
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================
# PROJECT ROOT
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# IMPORTS
# ============================================

from src.rag_engine import RAGEngine
from src.mcp_server import set_rag_engine
from src.agent import get_agent


# ============================================
# ENVIRONMENT
# ============================================

load_dotenv()


# ============================================
# STREAMLIT CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ConceptForge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown(
    """
    <style>

    /* ========================================
       GLOBAL
       ======================================== */

    .stApp {
        background-color: #f3f8f4;
    }

    .main {
        background-color: #f3f8f4;
    }

    /* ========================================
       HEADER
       ======================================== */

    .conceptforge-header {
        background: linear-gradient(
            135deg,
            #dcefe3 0%,
            #eef7f1 100%
        );

        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 25px;

        border: 1px solid #c8e1d1;

        box-shadow:
            0 4px 12px rgba(53, 94, 70, 0.06);
    }

    .conceptforge-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #234b35;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .conceptforge-subtitle {
        font-size: 1rem;
        color: #557363;
        margin-top: 6px;
    }

    .microscope {
        font-size: 2.5rem;
        margin-right: 12px;
    }

    /* ========================================
       ACTIVE DOCUMENT CARD
       ======================================== */

    .active-document {
        background-color: #ffffff;

        border-left: 5px solid #79a88b;

        padding: 14px 18px;
        border-radius: 10px;

        margin-bottom: 20px;

        box-shadow:
            0 2px 8px rgba(40, 70, 50, 0.05);
    }

    .active-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #789084;
        font-weight: 600;
    }

    .active-name {
        font-size: 1rem;
        font-weight: 600;
        color: #294d38;
        margin-top: 3px;
    }

    /* ========================================
       SIDEBAR
       ======================================== */

    section[data-testid="stSidebar"] {
        background-color: #e7f2ea;
        border-right: 1px solid #cddfd2;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #294d38;
    }

    /* ========================================
       DOCUMENT LIST
       ======================================== */

    .document-item {
        background-color: #f8fbf9;

        border: 1px solid #d5e5da;

        padding: 9px 12px;

        border-radius: 8px;

        margin-bottom: 6px;

        font-size: 0.85rem;

        color: #486353;
    }

    .document-active {
        background-color: #d9eddf;

        border: 1px solid #a9cdb4;

        color: #28523a;

        font-weight: 600;
    }

    /* ========================================
       CHAT
       ======================================== */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 10px;
    }

    /* ========================================
       BUTTONS
       ======================================== */

    .stButton > button {
        border-radius: 8px;

        border: 1px solid #9dbca7;

        background-color: #ffffff;

        color: #31553e;

        font-weight: 500;
    }

    .stButton > button:hover {
        border-color: #6f9d7d;
        color: #234b35;
        background-color: #edf6ef;
    }

    /* ========================================
       FILE UPLOADER
       ======================================== */

    [data-testid="stFileUploader"] {
        background-color: #f8fbf9;
        border-radius: 10px;
        padding: 6px;
    }

    /* ========================================
       STATUS
       ======================================== */

    [data-testid="stStatusWidget"] {
        border-radius: 10px;
        border: 1px solid #c9ddd0;
    }

    /* ========================================
       DIVIDERS
       ======================================== */

    hr {
        border-color: #d4e3d8;
    }

    /* ========================================
       FOOTER
       ======================================== */

    .conceptforge-footer {
        text-align: center;

        color: #81958a;

        font-size: 0.75rem;

        padding: 25px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# INITIALIZE RAG ENGINE
# ============================================

@st.cache_resource
def initialize_rag():
    """
    Initialize the persistent RAG engine.

    The RAG engine automatically restores the existing
    ChromaDB index and active document.
    """
    return RAGEngine()


rag_engine = initialize_rag()


# Share the same RAG engine instance with MCP.
set_rag_engine(rag_engine)


# ============================================
# INITIALIZE AGENT
# ============================================

@st.cache_resource
def initialize_agent():
    """
    Initialize the ConceptForge agent.

    MCP tool discovery is asynchronous.
    """
    return asyncio.run(get_agent())


# ============================================
# SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-user"

if "agent" not in st.session_state:
    st.session_state.agent = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# ============================================
# HEADER
# ============================================

st.markdown(
    """
    <div class="conceptforge-header">
        <div class="conceptforge-title">
            🔬 ConceptForge
        </div>
        <div class="conceptforge-subtitle">
            Intelligent Research Assistant for document analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================
# CAPABILITIES
# ============================================

st.markdown("### What can ConceptForge do?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **📄 Summarize documents**

        Get a concise overview of the active document.
        """
    )

with col2:
    st.markdown(
        """
        **🔎 Search documents**

        Find specific information using semantic search.
        """
    )

with col3:
    st.markdown(
        """
        **🧠 Generate concept maps**

        Visualize key concepts and their relationships.
        """
    )

st.divider()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.header("📚 Documents")

    documents = rag_engine.get_indexed_documents()
    active_document = rag_engine.get_active_document()

    # ----------------------------------------
    # CURRENT DOCUMENT
    # ----------------------------------------

    if active_document:

        st.markdown(
            "## Active document"
        )

        st.info(
            f"📄 {active_document}"
        )
        
    # ----------------------------------------
    # INDEXED DOCUMENTS
    # ----------------------------------------

    if documents:

        st.caption("Indexed documents")

        for document in documents:

            if document == active_document:

                st.markdown(
                    f"""
                    <div class="document-item document-active">
                        ● {document}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="document-item">
                        ○ {document}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.info(
            "No documents indexed yet."
        )

    st.divider()

    # ========================================
    # DOCUMENT UPLOAD
    # ========================================

    st.subheader("Add document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_file is not None:

        # Prevent repeated indexing after reruns.
        if (
            st.session_state.last_uploaded_file
            != uploaded_file.name
        ):

            upload_dir = PROJECT_ROOT / "uploads"
            upload_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_path = upload_dir / uploaded_file.name

            try:

                with open(file_path, "wb") as f:
                    f.write(
                        uploaded_file.getbuffer()
                    )

                with st.spinner(
                    "Indexing document..."
                ):

                    rag_engine.load_pdf_and_index(
                        str(file_path)
                    )

                st.session_state.last_uploaded_file = (
                    uploaded_file.name
                )

                st.success(
                    f"Loaded: {uploaded_file.name}"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error loading document: {e}"
                )

    # ========================================
    # ACTIVE DOCUMENT SELECTOR
    # ========================================

    documents = rag_engine.get_indexed_documents()
    active_document = rag_engine.get_active_document()

    if documents:

        st.subheader("Select document")

        selected_document = st.selectbox(
            "Active document",
            documents,
            index=(
                documents.index(active_document)
                if active_document in documents
                else 0
            ),
            label_visibility="collapsed",
        )

        if st.button(
            "Set as active",
            use_container_width=True,
        ):

            if rag_engine.set_active_document(
                selected_document
            ):

                # Make sure MCP uses the same instance.
                set_rag_engine(rag_engine)

                st.success(
                    f"Active document changed to:\n"
                    f"{selected_document}"
                )

                st.rerun()

            else:

                st.error(
                    "Could not change active document."
                )


# ============================================
# CREATE AGENT
# ============================================

if st.session_state.agent is None:

    try:

        st.session_state.agent = initialize_agent()

    except Exception as e:

        st.error(
            f"Could not initialize ConceptForge agent: {e}"
        )

        st.stop()


agent = st.session_state.agent


# ============================================
# DISPLAY CHAT HISTORY
# ============================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================
# ASYNC AGENT EXECUTION
# ============================================

async def run_agent(user_message: str):
    """
    Send a user message to the ConceptForge agent.

    MCP tools are asynchronous, so the agent is invoked
    using ainvoke().
    """

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": (
                    st.session_state.thread_id
                )
            }
        },
    )

    return response


# ============================================
# CHAT INPUT
# ============================================

user_input = st.chat_input(
    "Ask something about the active document..."
)


if user_input:

    # ----------------------------------------
    # SAVE USER MESSAGE
    # ----------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # ----------------------------------------
    # DISPLAY USER MESSAGE
    # ----------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)

    # ----------------------------------------
    # AGENT RESPONSE
    # ----------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.status(
                "🔬 ConceptForge is working...",
                expanded=True,
            ) as status:

                st.write(
                    "Connecting to the research agent..."
                )

                st.write(
                    f"Analyzing: "
                    f"{rag_engine.get_active_document()}"
                )

                st.write(
                    "Searching the indexed document..."
                )

                result = asyncio.run(
                    run_agent(user_input)
                )

                status.update(
                    label="✓ Analysis complete",
                    state="complete",
                    expanded=False,
                )

            # --------------------------------
            # EXTRACT ASSISTANT RESPONSE
            # --------------------------------

            messages = result.get(
                "messages",
                []
            )

            assistant_message = None

            for message in reversed(messages):

                if getattr(
                    message,
                    "type",
                    None
                ) != "ai":

                    continue

                content = getattr(
                    message,
                    "content",
                    None
                )

                if not content:
                    continue

                # Gemini may return content as a
                # list of structured blocks.
                if isinstance(content, list):

                    text_parts = []

                    for block in content:

                        if isinstance(
                            block,
                            dict
                        ):

                            if (
                                block.get("type")
                                == "text"
                            ):

                                text = block.get(
                                    "text",
                                    ""
                                )

                                if text:
                                    text_parts.append(
                                        text
                                    )

                        elif isinstance(
                            block,
                            str
                        ):

                            text_parts.append(
                                block
                            )

                    content = "\n".join(
                        text_parts
                    )

                assistant_message = str(
                    content
                )

                if assistant_message.strip():
                    break

            # --------------------------------
            # FALLBACK
            # --------------------------------

            if not assistant_message:

                assistant_message = (
                    "The agent did not return "
                    "a response."
                )

            # --------------------------------
            # DISPLAY RESPONSE
            # --------------------------------

            st.markdown(
                assistant_message
            )

            # --------------------------------
            # SAVE RESPONSE
            # --------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                }
            )

        except Exception as e:

            error_message = (
                f"Error communicating with the agent: {e}"
            )

            st.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


# ============================================
# FOOTER
# ============================================

st.markdown(
    """
    <div class="conceptforge-footer">
        ConceptForge · Intelligent document analysis
    </div>
    """,
    unsafe_allow_html=True,
)