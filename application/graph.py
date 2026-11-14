from langgraph.graph.message import add_messages
from typing import Dict, List, Any, TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from ai_model import ai_model

#################################################################################
# --- System Prompt ---
#################################################################################
SYSTEM_PROMPT = """\
You are an intelligent, helpful, and reliable AI assistant. Your goal is to assist users by providing clear, accurate, and structured answers.
"""

# --- Define the Graph ---

def build_graph():
    """
    Build the LangChain graph 
    """
    state_graph = StateGraph(MessagesState)
    
    model_ai = ai_model.get()
    
    def agent_node(state: MessagesState) ->  dict[str, Any]:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model_ai.invoke(messages)

        return {"messages": [response]}

    state_graph.add_node("agent", agent_node)

    # Graph flow
    state_graph.add_edge(START, "agent")
    state_graph.add_edge("agent", END)

    graph = state_graph.compile()

    return graph

