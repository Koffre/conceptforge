"""
Agent module for ConceptForge.

Connects the LangChain agent to the ConceptForge MCP server
and exposes the agent through an asynchronous interface.
"""

import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent
from langchain.messages import ToolMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================
# ENVIRONMENT
# ============================================

load_dotenv()


# ============================================
# STATE DEFINITION
# ============================================

class ConceptForgeState(AgentState):
    """
    State schema for the ConceptForge agent.
    """

    documents_indexed: bool = False
    current_document: Optional[str] = None


# ============================================
# CONTEXT DEFINITION
# ============================================

@dataclass
class UserContext:
    """
    Runtime context for the ConceptForge agent.
    """

    user_id: str = "default"
    preferred_language: str = "English"


# ============================================
# MIDDLEWARE
# ============================================

@before_agent
def trim_tool_messages(
    state: ConceptForgeState,
    runtime
) -> dict | None:
    """
    Remove ToolMessage objects from the conversation history.

    This keeps the conversation history smaller and avoids
    unnecessarily accumulating tool outputs.
    """

    messages = state.get("messages", [])

    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    if tool_messages:

        return {
            "messages": [
                RemoveMessage(id=message.id)
                for message in tool_messages
            ]
        }

    return None


# ============================================
# CREATE AGENT
# ============================================

async def create_conceptforge_agent():
    """
    Create and configure the ConceptForge agent.

    This function is asynchronous because MCP tool discovery
    is asynchronous.
    """

    # ----------------------------------------
    # LLM
    # ----------------------------------------

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )

    # ----------------------------------------
    # MCP CLIENT
    # ----------------------------------------

    mcp_client = MultiServerMCPClient(
        {
            "conceptforge": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "src.mcp_server"],
            }
        }
    )

    # ----------------------------------------
    # DISCOVER MCP TOOLS
    # ----------------------------------------

    tools = await mcp_client.get_tools()

    # ----------------------------------------
    # LOAD MCP PROMPT
    # ----------------------------------------

    prompt_result = await mcp_client.get_prompt(
        "conceptforge",
        "conceptforge_prompt",
    )

    # The MCP prompt returns a list of messages.
    # Extract the text from the first message.

    system_prompt = ""

    if prompt_result:

        first_message = prompt_result[0]

        if hasattr(first_message, "content"):

            content = first_message.content

            if isinstance(content, str):
                system_prompt = content

            elif isinstance(content, list):

                parts = []

                for item in content:

                    if hasattr(item, "text"):
                        parts.append(item.text)

                system_prompt = "\n".join(parts)

    # Fallback in case the MCP prompt is unavailable.
    if not system_prompt:

        system_prompt = """
        You are ConceptForge, an intelligent research assistant
        for document analysis.

        Always use the available MCP tools to answer questions
        about indexed documents.

        Respond in the same language as the user.

        Use the active document as the source of information.

        Available capabilities include:

        - semantic document search
        - document summarization
        - concept map generation
        - document listing
        - document selection

        Do not invent information that is not supported by
        the indexed document.
        """

    # ----------------------------------------
    # CREATE LANGCHAIN AGENT
    # ----------------------------------------

    agent = create_agent(
        model=model,
        tools=tools,
        state_schema=ConceptForgeState,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        middleware=[
            trim_tool_messages
        ],
        system_prompt=system_prompt,
    )

    return agent


# ============================================
# GLOBAL AGENT
# ============================================

_agent = None


# ============================================
# GET AGENT
# ============================================

async def get_agent():
    """
    Get the global ConceptForge agent.

    The agent is created only once and reused afterwards.
    """

    global _agent

    if _agent is None:

        _agent = await create_conceptforge_agent()

    return _agent