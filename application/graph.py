from langgraph.graph.message import add_messages
from typing import Dict, List, Any, TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from ai_model import ai_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

class Decision(BaseModel):
	approved: bool
	note: str | None = None

#################################################################################
# --- System Prompt ---
#################################################################################
SYSTEM_PROMPT = """\
You are an intelligent assistant connected to an Immich photo server.
Your role is to understand user requests in natural language and execute the appropriate actions on the Immich server.
You have access to the following tools:
    search_immich_assets: smart search (CLIP smart search) for images.
    ESSENTIAL RULE FOR SEARCH: The Immich server uses CLIP models based on the English language. ALWAYS translate search concepts from Romanian into clear English keywords before calling search_immich_assets (for example: „câine” / „câini” -> query="dog", „pisică” / „pisici” -> query="cat", „plajă / mare” -> query="beach", „mașină roșie” -> query="red car", „munte” -> query="mountain").
    create_immich_album: creates a new album on the Immich server (receives album_name, optionally description, and optionally the list of asset_ids to add directly into the album).
    get_immich_albums: lists all existing albums.
    add_assets_to_immich_album: adds photos (by asset ID) into an existing album.
    get_immich_asset: details about a specific asset based on its ID.
Instructions for compound requests (e.g., „fă un album separat și pune acolo o poză cu pisica / câinele”):
    Step 1: First, search for the appropriate photo using search_immich_assets with the term translated into English (e.g., "dog", "cat").
    Step 2: From the list of results returned by the tool, extract the ID of the relevant photo(s) (the id field).
    Step 3: Call create_immich_album, explicitly specifying both album_name and asset_ids containing the list of found ID(s) (e.g., asset_ids=["id_extras_din_cautare"]).
    STRICT RULE: If the user asked you to put photos into the album, NEVER call create_immich_album without the asset_ids parameter or with an empty list! Put the found photo ID directly into asset_ids.
    If the search yields no photos or returns an error, DO NOT create an empty album; inform the user politely in Romanian that you could not find matching photos.
    Respond to the user in a friendly and concise manner in Romanian.
"""

# Sensitive tools that require human confirmation
SENSITIVE_TOOLS = {"create_immich_album", "add_assets_to_immich_album"}

# Import Immich tools
from immich_tools import (
    immich_tools,
    search_immich_assets,
    get_immich_asset,
    create_immich_album,
    get_immich_albums,
    add_assets_to_immich_album
)

# --- Define the Graph ---
def build_graph():
    """
    Build the LangChain graph 
    """
    state_graph = StateGraph(MessagesState)
    
    model_ai = ai_model.get()
    
    def agent_node(state: MessagesState) ->  Command[Literal["human_confirmation", "tools"]]:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model_ai.invoke(messages)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            return Command(
                update = {"messages": [response]},
                goto = END
            )

        has_sensitive = any(tc["name"] in SENSITIVE_TOOLS for tc in response.tool_calls)
        goto = "human_confirmation" if has_sensitive else "tools"
        return Command(
            update = {"messages": [response]},
            goto = goto
        )

    def human_confirmation_node(state: MessagesState) -> Command[Literal["tools", "__end__"]]:
        """
        Human-in-the-loop decision node using native interrupt().
        - Pauses execution and waits for Command("approved": True / False).
        - If "approved" is False, it attaches tool cancellation messages and a predefined
          AIMessage directly, concluding the turn without an extra LLM call.
        - If "approved" is True, it forwards execution to the tools node.
        """
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        # Suspend node execution until resumed via Command(resume=...)
        status = interrupt({
            "action": "human_confirmation",
            "tool_calls": tool_calls},
            response_schema=Decision
        )

        if status.approved is not True:
            cancel_messages = [
                ToolMessage(
                    content="Action has been refused/cancelled by the user.",
                    tool_call_id=tc.get("id", "cancelled_call")
                )
                for tc in tool_calls
            ]
            response_message = AIMessage(
                content="Am anulat acțiunea la cererea ta. Nu a fost efectuată nicio modificare pe serverul Immich."
            )
            return Command(
                update={
                    "messages": cancel_messages + [response_message],
                },
                goto=END
            )

        return Command(
            goto="tools"
        )

    # Nodes
    tool_node = ToolNode(immich_tools)
    state_graph.add_node("tools", tool_node)
    state_graph.add_node("agent", agent_node)
    state_graph.add_node("human_confirmation", human_confirmation_node)

    # Graph flow
    state_graph.add_edge(START, "agent")
    state_graph.add_edge("tools", "agent")

    graph = state_graph.compile(checkpointer=MemorySaver())

    return graph

