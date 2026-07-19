"""
wealthdesk/state.py
-------------------
The shared state that flows through the LangGraph graph.

Session 3 adds query_type so the router can direct each question
to the right node (respond, escalate, or decline).
"""
from typing import TypedDict


class WealthDeskState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]

    # -----------------------------------------------------------------------
    # TODO 1 of 4 -- Add the query_type field
    # -----------------------------------------------------------------------
    # query_type is written by classify() and read by route_query().
    # Type hint: str
    # Valid values: "SIMPLE", "COMPLEX", "OUT_OF_SCOPE"
    #
    # TODO: add  query_type: str
