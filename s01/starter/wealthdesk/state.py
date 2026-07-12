"""
wealthdesk/state.py
-------------------
The shared state that flows through the LangGraph graph.

Every node reads from this state and writes back a partial update.
Only define the shape here -- no logic.
"""
from typing import TypedDict

class WealthDeskState(TypedDict):
    customer_message: str
    response: str
    history: list[dict[str, str]]  # List of dicts with keys "role" and "content"4
    query_type: str  # SIMPLE, COMPLEX, or OUT_OF_SCOPE

# Guard: raises at import time if the fields haven't been defined yet.
if "customer_message" not in WealthDeskState.__annotations__:
    raise NotImplementedError(
        "TODO 3: define 'customer_message: str' and 'response: str' "
        "in WealthDeskState in wealthdesk/state.py"
    )
