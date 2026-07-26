# src/mcp_server.py
"""
MCP Server for ConceptForge.

This module exposes RAG functionality as MCP tools:
- search_documents
- summarize_document
- generate_concept_map
- list_documents
"""

import os
import asyncio
from typing import Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from src.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# ============================================
# GLOBAL RAG ENGINE INSTANCE
# ============================================

_rag_engine: RAGEngine = None

def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


# ============================================
# MCP SERVER
# ============================================

# Create the MCP server
mcp = FastMCP("conceptforge-mcp")


@mcp.tool()
def search_documents(query: str) -> Dict[str, Any]:
    """
    Search the indexed documents for information relevant to the query.
    
    Args:
        query: The search query
        
    Returns:
        Dictionary with search results
    """
    engine = get_rag_engine()
    
    if not engine.vector_store:
        return {"error": "No documents have been indexed yet. Please upload a document first."}
    
    try:
        results = engine.search(query, k=3)
        
        if not results:
            return {"message": "No relevant information found in the document."}
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:500],
                "full_length": len(doc.page_content)
            })
        
        return {
            "query": query,
            "results_count": len(results),
            "results": formatted_results
        }
    
    except Exception as e:
        return {"error": f"Error searching documents: {str(e)}"}


@mcp.tool()
def summarize_document() -> str:
    """
    Generate a summary of the indexed document.
    
    Returns:
        Summary text
    """
    engine = get_rag_engine()
    
    if not engine.documents:
        return "No documents have been indexed yet. Please upload a document first."
    
    try:
        summary = engine.get_document_summary(k=5)
        
        if not summary:
            return "Could not generate a summary. The document may be empty."
        
        if len(summary) > 3000:
            summary = summary[:3000] + "... [truncated]"
        
        return f"📝 **Document Summary**\n\n{summary}"
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"


@mcp.tool()
def generate_concept_map() -> str:
    """
    Generate a concept map from the indexed document.
    
    Returns:
        Concept map in Mermaid format
    """
    engine = get_rag_engine()
    
    if not engine.documents:
        return "No documents have been indexed yet. Please upload a document first."
    
    # Get the document text
    document_text = "\n\n".join([doc.page_content for doc in engine.documents[:10]])
    
    # Import the LLM
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    prompt = f"""
    Analyze the following document text and create a concept map in Mermaid format.
    
    Document text:
    {document_text[:8000]}
    
    Instructions:
    1. Identify the key concepts (3-6 main concepts)
    2. Show the relationships between them
    3. Use Mermaid syntax (graph TD)
    4. Keep it clear and structured
    
    Format your response as a Mermaid code block.
    Use 'mermaid' as the language identifier.
    
    Example:
    mermaid
    graph TD
        A[Concept A] --> B[Concept B]
    
    Only respond with the Mermaid code block, no explanation.
    """
    
    try:
        response = model.invoke(prompt)
        return f"🧠 **Concept Map**\n\n{response.content}"
    
    except Exception as e:
        return f"Error generating concept map: {str(e)}"


@mcp.tool()
def list_documents() -> str:
    """
    List all currently indexed documents.
    
    Returns:
        List of document names
    """
    engine = get_rag_engine()
    
    if not engine.documents:
        return "No documents have been indexed yet."
    
    # Get unique source names
    sources = set()
    for doc in engine.documents:
        source = doc.metadata.get("source", "Unknown")
        if source != "Unknown":
            sources.add(os.path.basename(source))
    
    if sources:
        return f"📄 **Indexed Documents**\n\n" + "\n".join(f"• {s}" for s in sources)
    else:
        return "Documents are indexed but source information is not available."


@mcp.tool()
def load_document(file_path: str) -> str:
    """
    Load and index a PDF document.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Status message
    """
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    try:
        engine = get_rag_engine()
        engine.load_pdf_and_index(file_path)
        return f"✅ Document loaded and indexed successfully: {os.path.basename(file_path)}"
    
    except Exception as e:
        return f"❌ Error loading document: {str(e)}"


# ============================================
# MCP RESOURCES
# ============================================

@mcp.resource("conceptforge://status")
def get_status():
    """Get the current status of the RAG engine."""
    engine = get_rag_engine()
    doc_count = len(engine.documents) if engine.documents else 0
    
    status = f"""
    **ConceptForge MCP Server Status**
    
    - Documents indexed: {doc_count}
    - Vector store active: {'✅ Yes' if engine.vector_store else '❌ No'}
    - Chunk size: {engine.chunk_size}
    - Chunk overlap: {engine.chunk_overlap}
    """
    return status


# ============================================
# MCP PROMPT
# ============================================

@mcp.prompt()
def conceptforge_prompt():
    """
    System prompt for the ConceptForge agent.
    """
    return """
    You are ConceptForge, an intelligent research assistant.
    
    Your purpose is to help users understand and analyze documents.
    
    AVAILABLE TOOLS:
    1. **search_documents(query)** - Search the indexed document for specific information
    2. **summarize_document()** - Generate an overview of the document
    3. **generate_concept_map()** - Create a visual concept map from the document
    4. **list_documents()** - List all indexed documents
    5. **load_document(file_path)** - Load and index a new document
    
    GUIDELINES:
    - Always be helpful and professional
    - Use the tools when you need specific information
    - If you don't know something, say so honestly
    - Keep responses clear and well-structured
    
    When a user uploads a document, use load_document() to index it.
    """


# ============================================
# RUN THE SERVER
# ============================================

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")