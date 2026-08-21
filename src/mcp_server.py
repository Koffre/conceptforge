"""
MCP Server for ConceptForge.

This module exposes RAG functionality as MCP tools:
- search_documents
- summarize_document
- generate_concept_map
- list_documents
- select_document
- load_document
"""

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    try:
        from mcp_fastmcp import FastMCP
    except Exception:
        try:
            from fastmcp import FastMCP
        except Exception:
            logging.warning("Could not import FastMCP.")
            FastMCP = None

from src.rag_engine import RAGEngine

load_dotenv()

_rag_engine: Optional[RAGEngine] = None


def set_rag_engine(engine: RAGEngine) -> None:
    """Share a RAG Engine instance with the MCP Server."""
    global _rag_engine
    _rag_engine = engine


def get_rag_engine() -> RAGEngine:
    """Return the shared RAG Engine or create one if necessary."""
    global _rag_engine

    if _rag_engine is None:
        _rag_engine = RAGEngine()

    return _rag_engine


if FastMCP is None:
    raise ImportError(
        "FastMCP could not be imported. "
        "Please install the MCP package required by ConceptForge."
    )

mcp = FastMCP("conceptforge-mcp")


@mcp.tool()
def search_documents(query: str) -> Dict[str, Any]:
    """Search the currently active document."""
    engine = get_rag_engine()

    if not engine.has_index():
        return {"error": "No documents have been indexed yet. Please upload a document first."}

    active_document = engine.get_active_document()

    if not active_document:
        return {"error": "No active document is selected. Please select a document first."}

    try:
        results = engine.search(query, k=3)

        if not results:
            return {"message": "No relevant information found in the active document."}

        formatted_results = []

        for i, doc in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "source": doc.metadata.get(
                    "source_name",
                    os.path.basename(doc.metadata.get("source", "Unknown")),
                ),
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:500],
                "full_length": len(doc.page_content),
            })

        return {
            "query": query,
            "active_document": active_document,
            "results_count": len(results),
            "results": formatted_results,
        }

    except Exception as e:
        return {"error": f"Error searching active document: {str(e)}"}


@mcp.tool()
def summarize_document() -> str:
    """Generate an overview of the currently active document."""
    engine = get_rag_engine()

    if not engine.has_index():
        return "No documents have been indexed yet. Please upload a document first."

    active_document = engine.get_active_document()

    if not active_document:
        return "No active document is selected. Please select a document first."

    try:
        summary = engine.get_document_summary(k=5)

        if not summary:
            return "Could not generate a summary. The active document may be empty."

        if len(summary) > 3000:
            summary = summary[:3000] + "... [truncated]"

        return f"Document: {active_document}\n\n{summary}"

    except Exception as e:
        return f"Error generating summary: {str(e)}"


@mcp.tool()
def generate_concept_map() -> str:
    """Generate a Mermaid concept map from the active document."""
    engine = get_rag_engine()

    if not engine.has_index():
        return "No documents have been indexed yet. Please upload a document first."

    active_document = engine.get_active_document()

    if not active_document:
        return "No active document is selected. Please select a document first."

    try:
        documents = engine.get_document_chunks(k=10)

        if not documents:
            return "Could not retrieve document content from the persistent index."

        document_text = "\n\n".join(doc.page_content for doc in documents)

        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

        prompt = f"""
Analyze the following document text and create a concept map.

Document: {active_document}

Document text:
{document_text[:8000]}

Instructions:
1. Identify 3-6 main concepts.
2. Show the relationships between them.
3. Use valid Mermaid syntax.
4. Start with: graph TD
5. Return ONLY the Mermaid syntax.
6. Do not use Markdown code fences.
7. Do not add explanations before or after the graph.
"""

        response = model.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            content = "".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )

        content = str(content).strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        return f"Document: {active_document}\n\n{content}"

    except Exception as e:
        return f"Error generating concept map: {str(e)}"


@mcp.tool()
def list_documents() -> str:
    """List all documents in the persistent index and identify the active one."""
    engine = get_rag_engine()

    try:
        documents = engine.get_indexed_documents()

        if not documents:
            return "No documents have been indexed yet."

        active_document = engine.get_active_document()
        lines = ["Indexed Documents", ""]

        for document in documents:
            marker = " [ACTIVE]" if document == active_document else ""
            lines.append(f"- {document}{marker}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving indexed documents: {str(e)}"


@mcp.tool()
def select_document(document_name: str) -> str:
    """
    Select an already-indexed document as active.
    Does not upload, duplicate, or delete anything.
    """
    engine = get_rag_engine()

    try:
        documents = engine.get_indexed_documents()

        if not documents:
            return "No documents have been indexed yet."

        if document_name not in documents:
            return (
                f"Document not found: {document_name}\n\n"
                "Available documents:\n"
                + "\n".join(f"- {document}" for document in documents)
            )

        if engine.set_active_document(document_name):
            return f"Active document changed to: {document_name}"

        return f"Could not select document: {document_name}"

    except Exception as e:
        return f"Error selecting document: {str(e)}"


@mcp.tool()
def load_document(file_path: str) -> str:
    """
    Load and index a new PDF.

    The document is added to the persistent library and becomes active.
    Existing documents are not deleted.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    if not file_path.lower().endswith(".pdf"):
        return f"File must be a PDF: {file_path}"

    try:
        engine = get_rag_engine()
        engine.load_pdf_and_index(file_path)

        return (
            "Document loaded successfully.\n"
            f"Active document: {os.path.basename(file_path)}\n"
            f"Total indexed documents: {len(engine.get_indexed_documents())}\n"
            f"Total indexed chunks: {engine.get_indexed_chunk_count()}"
        )

    except Exception as e:
        return f"Error loading document: {str(e)}"


@mcp.resource("conceptforge://status")
def get_status() -> str:
    """Return the current ConceptForge RAG status."""
    engine = get_rag_engine()

    has_index = engine.has_index()

    try:
        indexed_documents = engine.get_indexed_documents()
    except Exception:
        indexed_documents = []

    active_document = engine.get_active_document()

    return (
        "ConceptForge MCP Server Status\n"
        "--------------------------------\n"
        f"Persistent index available: {'Yes' if has_index else 'No'}\n"
        f"Indexed documents: {len(indexed_documents)}\n"
        f"Active document: {active_document or 'None'}\n"
        f"Vector store active: {'Yes' if engine.vector_store else 'No'}\n"
        f"Total chunks: {engine.get_indexed_chunk_count()}\n"
        f"Chunk size: {engine.chunk_size}\n"
        f"Chunk overlap: {engine.chunk_overlap}"
    )


@mcp.prompt()
def conceptforge_prompt() -> str:
    """System prompt for the ConceptForge agent."""
    return """
You are ConceptForge, an intelligent research assistant for document analysis.

DOCUMENT MODEL

ConceptForge maintains a persistent library of indexed documents.

Several documents may be stored at the same time, but only ONE document is
active at a time.

All document analysis operations work ONLY on the active document.

AVAILABLE CAPABILITIES

1. Semantic Search
   Search the active document for specific information.

2. Document Summarization
   Generate an overview of the active document.

3. Concept Mapping
   Generate a Mermaid concept map from the active document.

4. Document Library
   List all documents stored in the persistent index.

5. Document Selection
   Select an already-indexed document as the active document.

6. Document Loading
   Load a new PDF into the persistent library. The newly loaded document
   becomes active automatically.

IMPORTANT RULES

- Always use the available tools to answer questions about documents.
- Search, summaries, and concept maps must use the ACTIVE document only.
- If the user asks about another document already in the library, use
  list_documents() and select_document() to switch to it.
- Do NOT ask the user to upload a document that is already indexed.
- Loading a new document does NOT delete previously indexed documents.
- If no active document exists, ask the user to select or load one.
- If a tool fails, explain the error clearly.
- Respond in the same language as the user's question.

AVAILABLE TOOLS

search_documents(query)
Search the active document.

summarize_document()
Summarize the active document.

generate_concept_map()
Create a concept map from the active document.

list_documents()
List all documents stored in the persistent library and identify the active one.

select_document(document_name)
Change the active document to an already-indexed document.

load_document(file_path)
Load a new PDF and make it the active document.

RESPONSE GUIDELINES

Be clear and professional.

When referring to document information, identify the active document when useful.

If the user asks something outside the available document content, explain that
the answer must be based on the active document.

For concept maps, return the Mermaid graph in a format suitable for rendering.
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
