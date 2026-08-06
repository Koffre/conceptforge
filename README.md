# ⚙️ ConceptForge

**ConceptForge** is an intelligent research assistant that combines **RAG (Retrieval-Augmented Generation)** with **MCP (Model Context Protocol)** to help you analyze PDF documents, extract key insights, and visualize knowledge through concept maps.

---

## 🎯 Features

- 📄 **Document Upload** – Load and index PDF documents for analysis
- 🔍 **Semantic Search** – Find relevant information using natural language
- 📝 **Smart Summarization** – Generate concise and structured document summaries
- 🧠 **Concept Mapping** – Visualize key ideas and their relationships in Mermaid format
- 💬 **Conversational AI** – Interact with your documents through a chat interface
- 🌍 **Multi-language support** – Interface available in English and Spanish

---

## 🏗️ Architecture

User → Streamlit Interface → LangChain Agent → MCP Server → RAG Engine → Document Index


### Components

| Component       | Technology                   | Purpose                                        |
|-----------------|------------------------------|------------------------------------------------|
| **RAG Engine**  | LangChain, Gemini Embeddings | Document processing and semantic search        |
| **MCP Server**  | FastMCP                      | Exposes tools via Model Context Protocol       |
| **Agent**       | LangChain                    | Orchestrates conversation and tool usage       |
| **Interface**   | Streamlit                    | Web-based chat interface for users             |

---

## 🛠️ Technologies

- [LangChain](https://github.com/langchain-ai/langchain) — Framework for building LLM-powered applications
- [LangGraph](https://github.com/langchain-ai/langgraph) — State management for multi-agent systems
- [Google Gemini](https://ai.google.dev/gemini-api) — LLM for embeddings and generation
- [Streamlit](https://streamlit.io/) — Web interface for the chat application
- [uv](https://github.com/astral-sh/uv) — Fast Python package management
- [MCP](https://github.com/modelcontextprotocol) — Model Context Protocol for tool integration

---

## 📸 Screenshots

> **Screenshots coming soon.**  
> You can see the interface in action by running the application locally.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API Key
- `uv` (optional but recommended)

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Koffre/conceptforge.git
   cd conceptforge

2. Create and activate a virtual environment
uv venv
source .venv/Scripts/activate  # On Windows (Git Bash)

3. Install dependencies
uv pip install -r requirements.txt

4. Configure environment variables
Create a .env file in the root directory and add your Google Gemini API key:
GOOGLE_API_KEY=your_google_api_key_here

## 💻 Usage
Start the application
streamlit run src/main.py

Then open http://localhost:8501 in your browser.

## Example Workflow
Upload a PDF document
Use the sidebar to upload a file. The system will automatically index it.

Ask questions

resume el documento – Get a detailed summary

genera un mapa conceptual – Generate a concept map

busca información sobre [tema] – Search for specific information

summarize the document – Same commands work in English

Explore insights
The agent will respond with relevant information, summaries, or Mermaid diagrams.

## 📁 Project Structure

conceptforge/

├── src/

│   ├── __init__.py          # Package exports

│   ├── rag_engine.py        # RAG Engine (PDF loading, splitting, embeddings)

│   ├── mcp_server.py        # MCP Server (tools exposed via MCP)

│   ├── agent.py             # Agent configuration and middleware

│   └── main.py              # Streamlit interface entry point

├── samples/                 # Sample PDF documents

├── docs/                    # Documentation and screenshots (optional)

├── .env                     # Environment variables (not in repo)

├── .gitignore               # Ignored files

├── README.md                # Project documentation

└── requirements.txt         # Python dependencies


## 🔧 Configuration
Create a .env file with your API keys:

GOOGLE_API_KEY=your_google_api_key_here


## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.


## Author

Joffre Sanchez  - https://github.com/Koffre


## 🙏 Acknowledgments

- LangChain community for the excellent framework

- Google for Gemini API

- Open-source community for the tools and libraries

## 📬 Contact
If you have questions, suggestions, or feedback, feel free to open an issue or reach out via GitHub
