"""
wealthdesk/nodes.py
-------------------
Graph nodes and routing function for WealthDesk.

Session 3 adds classify() and route_query() so queries are directed
to respond(), escalate(), or decline() based on their type.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE, SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import classifier_llm, llm

# Phrases that should never reach the LLM — short-circuit before spending an API token.
# Provided: this is a defensive guardrail, not part of the routing exercise.
BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]


def classify(state: WealthDeskState) -> dict:
    """Classify the customer question into SIMPLE, COMPLEX, or OUT_OF_SCOPE.

    Note: classify only the current question -- do NOT include history here.
    """
    msg = state["customer_message"].strip()

    # Provided: input validation and pre-filter — cheap checks before the LLM call
    if not msg or len(msg) < 3 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE"}
    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE"}

    # -----------------------------------------------------------------------
    # TODO 2 of 4 -- Implement classify()
    # -----------------------------------------------------------------------
    # 1. Build the message list for the classifier LLM:
    #      messages = [
    #          SystemMessage(content=CLASSIFY_SYSTEM),
    #          HumanMessage(content=msg),
    #      ]
    #
    # 2. Call classifier_llm.invoke(messages) inside a try/except:
    #      On success:
    #        query_type = result.content.strip().upper()
    #        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
    #            query_type = "SIMPLE"   # safe default for unexpected output
    #      On exception:
    #        print(f"[WealthDesk] Classification error: {e}")
    #        query_type = "SIMPLE"       # safe default on failure
    #
    # 3. Return {"query_type": query_type}
    #    (This node only returns query_type; other fields are unchanged.)
    # -----------------------------------------------------------------------
    # TODO: implement this node
    pass


def respond(state: WealthDeskState) -> dict:
    """Handle SIMPLE queries. Provided -- no changes needed."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result        = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[WealthDesk] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: WealthDeskState) -> dict:
    """Handle COMPLEX queries with a canned RM referral. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: WealthDeskState) -> dict:
    """Handle OUT_OF_SCOPE queries with a canned decline. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: WealthDeskState) -> str:
    """Read query_type and return the name of the next node.

    LangGraph calls this after classify() runs. The string returned
    must match an existing node name in the graph.
    """
    # -----------------------------------------------------------------------
    # TODO 3 of 4 -- Implement route_query()
    # -----------------------------------------------------------------------
    # Read the query type and route accordingly:
    #   if query_type == "COMPLEX":      return "escalate"
    #   if query_type == "OUT_OF_SCOPE": return "decline"
    #   otherwise:                        return "respond"
    #
    # Use state.get("query_type", "SIMPLE") to read safely.
    # -----------------------------------------------------------------------
    # TODO: implement this function
    pass
