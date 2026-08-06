# src/agent.py
"""
Agent module for ConceptForge.
Connects to the MCP server and uses its tools.
"""

import asyncio
import os
from typing import Optional
from dataclasses import dataclass

from langchain.agents import create_agent, AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import before_agent
from langchain.messages import ToolMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

load_dotenv()


# ============================================
# STATE DEFINITION
# ============================================

class ConceptForgeState(AgentState):
    """State schema for the ConceptForge agent."""
    documents_indexed: bool = False
    current_document: Optional[str] = None


# ============================================
# CONTEXT DEFINITION
# ============================================

@dataclass
class UserContext:
    """User context for the agent."""
    user_id: str = "default"
    preferred_language: str = "English"


# ============================================
# MIDDLEWARE: Trim Tool Messages
# ============================================

@before_agent
def trim_tool_messages(state: ConceptForgeState, runtime) -> dict:
    """
    Remove ToolMessage objects from the conversation history.
    This keeps the conversation clean and reduces token usage.
    """
    messages = state.get("messages", [])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_messages:
        return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}
    return None


# ============================================
# CREATE THE AGENT
# ============================================

def create_conceptforge_agent():
    """Create and configure the ConceptForge agent."""
    
    # Create the model
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

    # Connect to the MCP server
    mcp_client = MultiServerMCPClient(
        {
            "conceptforge": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "src.mcp_server"],
            }
        }
    )
    
    # Discover tools from the MCP server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        tools = loop.run_until_complete(mcp_client.get_tools())
        prompt_result = loop.run_until_complete(mcp_client.get_prompt("conceptforge", "conceptforge_prompt"))
        system_prompt = prompt_result[0].content
    finally:
        loop.close()
        asyncio.set_event_loop(None)
        
    
    # Create the agent
    agent = create_agent(
        model=model,
        tools=tools,
        state_schema=ConceptForgeState,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        middleware=[trim_tool_messages],
        system_prompt=system_prompt
    )
    
    return agent


# ============================================
# GLOBAL AGENT INSTANCE
# ============================================

_agent = None

def get_agent():
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = create_conceptforge_agent()
    return _agent

