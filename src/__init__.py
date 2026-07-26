# src/__init__.py
"""
ConceptForge - Intelligent Research Assistant.

A multi-agent system for document analysis, semantic search, 
and concept mapping using RAG, MCP, and LangChain.
"""

from src.rag_engine import RAGEngine
from src.mcp_server import mcp
from src.agent import create_conceptforge_agent, get_agent

__version__ = "0.1.0"
__all__ = [
    "RAGEngine",
    "mcp",
    "create_conceptforge_agent",
    "get_agent",
]