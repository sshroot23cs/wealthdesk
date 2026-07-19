"""
wealthdesk/agent.py
-------------------
Graph construction and the terminal loop.

Run the agent from the repo root:
    cd cohort-1/wealthdesk/s01/starter
    python -m wealthdesk.agent

Session 1 graph:
    START --> respond --> END
"""
import os
import sqlite3 
from uuid import uuid4 

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .nodes import respond, classify, escalate, decline, retrieve_docs, route_query
from .state import WealthDeskState
from .config import CHECKPOINT_DB, ESCALATE_RESPONSE

def build_graph(checkpointer=None):
    """Build and compile the WealthDesk LangGraph graph."""
    try:
        builder = StateGraph(WealthDeskState)
        # START -> Classify -> route query -> respond / escalate / decline -> end
        # start node classify
        # then either of respond, escalate, decline node
        # then end node
        builder.add_node("classify", classify)
        builder.add_node("retrieve_docs", retrieve_docs)
        builder.add_node("decline", decline)
        builder.add_node("respond", respond)
        builder.set_entry_point("classify")
        builder.add_conditional_edges("classify", route_query, {
            "retrieve_docs": "retrieve_docs",
            "decline": "decline"
        })
        builder.add_edge("retrieve_docs", "respond")
        builder.add_edge("decline", END)
        builder.add_edge("respond", END)
        if checkpointer is None:
            checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)
    except Exception as e:
        print(f"[WealthDesk] Error building graph: {e}")
        raise


# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    """Run the WealthDesk agent in a terminal loop."""
    # Connect to the SQLite database for persistent checkpoints
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    # Build a new graph instance with the SQLite checkpointer
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}
    print("=" * 55)
    print("  WealthDesk | Bharat National Bank")
    print("  Type 'quit' to exit")
    print("  Session ID:", thread_id)
    print("=" * 55)
    # sanity check -- confirms config actually reached graph.invoke()
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "batch1-wealthdesk")
        print(f"  Tracing : LangSmith ({project})")
    print("=" * 55)
    

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nWealthDesk: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nWealthDesk: Thank you for choosing Bharat National Bank. Goodbye!")
            break

        # "response": "" is a placeholder to satisfy the TypedDict contract.
        # respond() overwrites it; graph.invoke() returns the full merged state.
        result = _graph.invoke({"customer_message": user_input, "response": ""}, config=config)
        route = result.get("query_type", "?")
        retrieved_docs = result.get("retrieved_docs", [])
        response = result.get("response", [])
        if retrieved_docs and response != ESCALATE_RESPONSE:
            print(f"\n[Retrieved {len(retrieved_docs)} docs]")
            # print the content of the retrieved docs for debugging
            print(f"\n[Retrieved docs content]")
            for i, doc in enumerate(retrieved_docs, start=1):
                print(f"\n[Doc {i}]\n{doc}")

        else:
            print(f"\n[No relevant docs retrieved]")
        print(f"\n[Routed: {route}]")
        print(f"\nWealthDesk: {result['response']}")


if __name__ == "__main__":
    run()
