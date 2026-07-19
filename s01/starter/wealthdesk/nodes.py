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

from .config import SYSTEM_PROMPT, CLASSIFY_SYSTEM_PROMPT, ESCALATE_RESPONSE, DECLINE_RESPONSE
from .state import WealthDeskState
from .tools import llm, classifer_llm

BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]

def classify(state: WealthDeskState) -> dict:
    """Classify the customer message and decide which node to invoke next."""
    
    msg = state["customer_message"].strip()
 
    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE"}
 
    if not msg or len(msg) < 10 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE"}
    
    message = state["customer_message"].lower()
    # use llm_classify llm model to classify the customer intent into SIMPLE, COMPLEX, or OUT_OF_SCOPE
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=message)
    ]

    # # Add the history to the messages list
    # history = state.get("history", [])
    # if history:
    #     for turn in history:
    #         if turn["role"] == "user":
    #             messages.append(HumanMessage(content=turn["content"]))
    #         elif turn["role"] == "assistant":
    #             messages.append(AIMessage(content=turn["content"]))
    
    try:
        result = classifer_llm.invoke(messages)
        query_type = result.content.strip().upper()
        
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            print(f"[WealthDesk] Unexpected classification result: {query_type}")
            query_type = "SIMPLE"  # default to SIMPLE if unexpected result
        
        # history.append({"role": "user", "content": state["customer_message"]})
        # history.append({"role": "assistant", "content": f"Classification: {query_type}"})

    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "SIMPLE"
  
    return {"query_type": query_type}
    # return {"query_type": "out_of_scope", "history": history}

def escalate(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}
 
def decline(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}
 
def route_query(state: WealthDeskState)->str:
   query_type = state.get("query_type","SIMPLE")
   if query_type == "COMPLEX":
      return "escalate"
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "respond"

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
