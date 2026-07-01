In Phase 1 I wrote a while True loop manually — every step coded by hand, every decision an if/else inside the loop. It worked, but it was fragile, hard to debug, and impossible to pause mid-execution. Phase 2 replaces that loop with a StateGraph — a directed graph where nodes are functions and edges are decisions. The agent node calls Claude, the routing function checks if tools are needed, the tool node runs them, and an unconditional edge loops back to the agent. Same 7 steps — different architecture. What changed is what I get for free: LangSmith traced every node automatically without a single print statement, the MemorySaver checkpointer saved state after every node so the graph can resume from any point if it crashes, and the conditional edge replaced my manual stop_reason check with an explicit routing function that anyone reading the code can understand immediately. The LangSmith trace I just saw — input, tool call, tool result, output — is my audit trail, my debug tool, and my proof of work all in one. In production this trace is how we find bugs, prove correctness to clients, and monitor agent behaviour at scale. That's the shift from Phase 1 to Phase 2 — from a loop I manage manually to a system that manages itself and shows me everything it's doing.

# Phase 2 Week 1 — LangGraph Foundations

## What changed from Phase 1
The while True loop is replaced by a StateGraph. Same 7 steps — different architecture. Nodes are functions, edges are decisions, state is the shared whiteboard that flows through everything.

## Key concepts
- KhopfaState TypedDict — shared whiteboard, all nodes read and write to it
- Annotated[list, operator.add] — messages accumulate, never overwritten
- agent_node — calls Claude, returns updated messages
- tool_node — executes tools, returns results
- should_continue — routing function, replaces if/else in Phase 1 loop
- MemorySaver — checkpointer, saves state after every node
- thread_id — unique conversation identifier, enables resumable execution
- interrupt_before — pauses graph before specified node, waits for human approval

## What LangGraph gives me that Phase 1 didn't
- LangSmith traces every node automatically — no print statements needed
- Durable execution — crash at node 5, resume from node 5
- Human-in-the-loop — pause before irreversible actions
- Clean branching — conditional edges replace messy if/else in loop

## Known issue
Human-in-the-loop interrupt fires before all booking details are collected. Fix: add validation check in should_continue — only route to confirm when name, phone, vehicle, location and price are all present in state.