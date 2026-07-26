# ⚙️ ConceptForge

**ConceptForge** is an intelligent research assistant that combines RAG (Retrieval-Augmented Generation) with MCP (Model Context Protocol) to help you analyze documents, extract insights, and visualize knowledge.

## 🎯 Features

- 📄 **Document Upload** – Load PDF documents for analysis
- 🔍 **Semantic Search** – Find relevant information using natural language
- 📝 **Smart Summarization** – Generate concise document summaries
- 🧠 **Concept Mapping** – Visualize key concepts and relationships in Mermaid format
- 💬 **Conversational AI** – Interact with your documents through a chat interface


## 🏗️ Architecture

User → Streamlit Interface → LangChain Agent → MCP Server → RAG Engine → Document Index

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **RAG Engine** | LangChain, Gemini Embeddings | Document processing and semantic search |
| **MCP Server** | FastMCP | Exposes tools via Model Context Protocol |
| **Agent** | LangChain | Orchestrates conversation and tool usage |
| **Interface** | Streamlit | Web-based chat interface |

## Technologies

- [LangChain](https://github.com/langchain-ai/langchain) — Framework for building LLM-powered applications
- [LangGraph](https://github.com/langchain-ai/langgraph) — State management for multi-agent systems
- [Google Gemini](https://ai.google.dev/gemini-api) — LLM for embeddings and generation
- [Streamlit](https://streamlit.io/) — Web interface for the chat application
- [uv](https://github.com/astral-sh/uv) — Fast Python package management

### Installation

## Prerequisites

- Python 3.12+
- Google Gemini API Key
- uv (optional but recommended)

### Setup

1. **Clone the repository:**
   
   git clone https://github.com/Koffre/conceptforge.git

2. Create a virtual environment: `uv venv`
   
3. Activate the environment: `source .venv/Scripts/activate`
   
5. Install dependencies: `uv pip install -r requirements.txt`
   
6. Configure environment variables: 
   Create a .env file in the root directory and add your key

## Usage

### Start the application
streamlit run src/main.py
Open http://localhost:8501 in your browser.

### Example Workflow

1. Upload a PDF document

2. Wait for indexing (automatic)

3. Ask questions:

  "What is the main topic of this document?"

  "Summarize the key findings"

  "Generate a concept map"

  "Find information about [specific topic]"


## Project Structure

conceptforge/
├── src/

│   ├── __init__.py          # Package exports

│   ├── rag_engine.py        # RAG Engine

│   ├── mcp_server.py        # MCP Server

│   ├── agent.py             # Agent configuration

│   └── main.py              # Streamlit interface

├── samples/                 # Sample PDFs

├── .env                     # Environment variables

├── .gitignore               # Ignored files

├── README.md                # Project documentation

└── requirements.txt         # Python dependencies


## 🔧 Configuration
Create a .env file with your API keys:

GOOGLE_API_KEY=your_google_api_key_here

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.


##📄 License
This project is licensed under the MIT License.


## Author

Joffre Sanchez  - https://github.com/Koffre


## 🙏 Acknowledgments

- LangChain community for the excellent framework

- Google for Gemini API

- Open-source community for the tools and libraries
