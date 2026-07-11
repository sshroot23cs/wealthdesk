"""
wealthdesk/nodes.py
-------------------
Node functions for the WealthDesk graph.

Each node is a plain Python function:
  - Input : the full WealthDeskState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core import messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import llm


def respond(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    messages = [
           SystemMessage(content=SYSTEM_PROMPT)
       ]
    
    # Add History to the messages list
    history = state.get("history", [])
    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
    # Append the new customer message to the messages list
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
      result = llm.invoke(messages)
      # Update the history with the new turn
      history.append({"role": "user", "content": state["customer_message"]})
      history.append({"role": "assistant", "content": result.content}) 
      return {"response": result.content, "history": history}
    except Exception as e:
      print(f"[WealthDesk] LLM error: {e}")
      return {"response": "Sorry, I encountered an error. Please try again."}


