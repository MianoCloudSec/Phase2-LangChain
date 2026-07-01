# langgraph_agent.py - Phase 2 Week 1
# Khopfa Towing agent rebuilt on LangGraph
# Replaces the raw while loop with a StateGraph
# Key concepts: State, Nodes, Edges, Conditional Routing, Checkpointing
#
# Mental models used:
#   1. StateGraph — nodes connected by edges, state flows through
#   2. Conditional edges — routing function decides next node
#   3. Checkpointer — durable execution, resumable from any point

import os
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from anthropic import Anthropic
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Brain and search clients — same as Phase 1
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# ── STATE ──────────────────────────────────────────────────────────────────
# KhopfaState is the shared whiteboard — every node reads and writes to it
# Why TypedDict: gives us type hints so we know what each field should contain
# Why Annotated[list, operator.add]: messages accumulate — never overwritten
# All other fields are overwritten by the last node that sets them

class KhopfaState(TypedDict):
    # Conversation history — accumulates across all nodes
    messages: Annotated[list, operator.add]
    
    # Customer details — collected by collect_info_node
    customer_name: str
    phone: str
    distance_km: float
    after_hours: bool
    vehicle: str
    location: str
    
    # Booking result — set by calculate_price_node
    price: str
    
    # Confirmation — set by confirm_booking_node
    confirmed: bool

# ── TOOLS ──────────────────────────────────────────────────────────────────
# Same tools as Phase 1 — tools are independent of the orchestration layer
# Swapping while loop for LangGraph changes nothing about the actual tools

def web_search(query: str) -> str:
    # Real Tavily search — clean web results for the model
    try:
        results = tavily.search(query=query, max_results=3)
        output = []
        for r in results.get("results", []):
            output.append(f"Source: {r['url']}\n{r['content'][:300]}")
        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

def calculate_tow_price(distance_km: float, after_hours: bool = False) -> str:
    # Minimum distance — under 10km returns base fee R350
    if distance_km < 10:
        price = 350
    else:
        base = 350
        per_km = 12
        price = base + (distance_km - 10) * per_km
    # After hours — 30% surcharge for night and weekend
    if after_hours:
        price = price * 1.3
    return f"R{price:.0f}"

# ── TOOL SCHEMAS ───────────────────────────────────────────────────────────
# Same Anthropic format as Phase 1
# These are passed to the model so it knows what tools are available

tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information about towing regulations, prices, or anything the customer asks about that requires current data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_tow_price",
        "description": "Calculate exact towing price for Khopfa Towing. ALWAYS use this before quoting any price. Never guess. Pass after_hours as true if customer mentions night, evening, after 7pm, Sunday or weekend.",
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_km": {
                    "type": "number",
                    "description": "Distance in kilometres"
                },
                "after_hours": {
                    "type": "boolean",
                    "description": "True if after 7pm or weekend, False otherwise"
                }
            },
            "required": ["distance_km"]
        }
    }
]

# ── NODES ──────────────────────────────────────────────────────────────────
# Each node takes state, does one job, returns only what changed
# Why one job per node: clean, testable, easy to debug in LangSmith
# Why return only changes: LangGraph merges return value into existing state

def agent_node(state: KhopfaState) -> dict:
    # Main agent node — calls Claude with current conversation
    # Why: model decides what to do next — call a tool or respond to customer
    # Returns: updated messages list with model's response
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="You are a towing assistant for Khopfa Towing in Limpopo. ALWAYS call calculate_tow_price before quoting any price. Never assume distance — always ask. Collect name, phone, vehicle, location and distance before confirming any booking.",
        messages=state["messages"],
        tools=tools
    )
    
    # Convert response to serialisable format
    new_messages = []
    
    if response.stop_reason == "end_turn":
        # Model finished — extract text
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        new_messages.append({"role": "assistant", "content": text})
    
    elif response.stop_reason == "tool_use":
        # Model wants tools — save full content for tool_node
        content = []
        for block in response.content:
            if hasattr(block, "text"):
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        new_messages.append({"role": "assistant", "content": content})
    
    return {
        "messages": new_messages,
        "_stop_reason": response.stop_reason
    }

def tool_node(state: KhopfaState) -> dict:
    # Tool execution node — runs tools requested by agent_node
    # Why separate node: clean separation of concerns, visible in LangSmith
    # Returns: tool results appended to messages
    
    # Find the last assistant message with tool calls
    last_message = state["messages"][-1]
    content = last_message.get("content", [])
    
    tool_results = []
    
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_name = block["name"]
            tool_input = block["input"]
            
            # Execute the tool
            try:
                if tool_name == "web_search":
                    result = web_search(**tool_input)
                elif tool_name == "calculate_tow_price":
                    result = calculate_tow_price(**tool_input)
                    # Update price in state
                else:
                    result = f"Unknown tool: {tool_name}"
            except Exception as e:
                result = f"Tool error: {str(e)}"
            
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": str(result)
            })
    
    # Send all tool results back as one user message
    new_message = {"role": "user", "content": tool_results}
    
    return {"messages": [new_message]}

# ── ROUTING ────────────────────────────────────────────────────────────────
# Conditional edge — reads state, returns name of next node
# Why: replaces the if/else inside the Phase 1 while loop
# Clean, explicit, visible in LangSmith as a decision point

def should_continue(state: KhopfaState) -> str:
    # Check the last message to decide where to go next
    last_message = state["messages"][-1]
    content = last_message.get("content", [])
    
    # If last message has tool_use blocks — go to tool_node
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return "tools"
    
    # Otherwise model is done — end the graph
    return "end"


# ── GRAPH CONSTRUCTION ─────────────────────────────────────────────────────
# Build the StateGraph — add nodes, add edges, compile
# This replaces the while True loop from Phase 1
# Why graph: explicit flow, durable execution, human-in-the-loop ready

def build_graph():
    # Create graph with KhopfaState
    graph = StateGraph(KhopfaState)
    
    # Add nodes — each is a focused function
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # Entry point — always start at agent node
    graph.set_entry_point("agent")
    
    # Conditional edge from agent — go to tools or end
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # model wants tools
            "end": END         # model is done
        }
    )
    
    # After tools always go back to agent
    # Why: same as Phase 1 Step 7 — loop back after tool execution
    graph.add_edge("tools", "agent")
    
    # Compile with checkpointer — durable execution enabled
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ── ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()
    
    # Thread ID — unique identifier for this conversation
    # Why: checkpointer uses this to save and resume state
    config = {"configurable": {"thread_id": "khopfa-test-001"}}
    
    # Initial state — empty messages, defaults for other fields
    initial_state = {
        "messages": [{"role": "user", "content": "Hi, how much to tow my car 25km in Polokwane?"}],
        "customer_name": "",
        "phone": "",
        "distance_km": 0.0,
        "after_hours": False,
        "vehicle": "",
        "location": "",
        "price": "",
        "confirmed": False
    }
    
    result = app.invoke(initial_state, config=config)
    
    # Print last message
    last = result["messages"][-1]
    if isinstance(last.get("content"), str):
        print(last["content"])
    else:
        print(last)