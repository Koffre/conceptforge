# ConceptForge

**ConceptForge** is an intelligent research assistant that transforms documents into structured knowledge. It combines Retrieval-Augmented Generation (RAG) with AI-powered concept mapping to help you read, understand, and synthesize information.

## Features

- **Document Upload**: Upload PDF files for analysis
- **Semantic Search**: Find relevant information using natural language queries
- **Smart Summaries**: Generate concise summaries of documents
- **Concept Mapping**: Automatically extract key concepts and visualize relationships
- **Chat Interface**: Interact with your documents through a user-friendly chat

## Architecture

ConceptForge is built using a multi-agent system architecture:

- **RAG Engine**: Indexes documents for semantic search using embeddings
- **Research Agent**: Handles queries and document interactions
- **Concept Mapper**: Extracts concepts and relationships from documents
- **Orchestrator**: Coordinates between agents and manages conversation state

## Technologies

- [LangChain](https://github.com/langchain-ai/langchain) — Framework for building LLM-powered applications
- [LangGraph](https://github.com/langchain-ai/langgraph) — State management for multi-agent systems
- [Google Gemini](https://ai.google.dev/gemini-api) — LLM for embeddings and generation
- [Streamlit](https://streamlit.io/) — Web interface for the chat application
- [uv](https://github.com/astral-sh/uv) — Fast Python package management

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) for Python package management
- Google Gemini API key

### Setup

1. **Clone the repository:**
   
   git clone https://github.com/Koffre/conceptforge.git

2. Crea un entorno virtual: `uv venv`
   
3. Activa el entorno: `source .venv/Scripts/activate`
   
5. Instala las dependencias: `uv pip install -r requirements.txt`
   
7. Configura tu `.env` con la `GOOGLE_API_KEY`  
   
## Project Structure

conceptforge/
├── src/
│   ├── __init__.py
│   ├── rag_engine.py        # RAG Engine
│   ├── mcp_server.py        # MCP Server
│   ├── agent.py             # Agent
│   └── main.py   
├── samples/
│   └── Web incentives in Ghana - Meuleman, Langer & Blom 2017 SSCR.pdf
├── temp/                    # Temporary file storage
├── .env                     # Environment variables (not in repo)
├── .gitignore               # Ignored files
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies


## Future Enhancements

- Support for additional document formats (DOCX, TXT, HTML)

- Multi-document analysis and comparison

- Export concept maps as images or markdown

- Integration with Tavily for web search

- Persistent vector store for long-term memory


## Author

Joffre Sanchez 
