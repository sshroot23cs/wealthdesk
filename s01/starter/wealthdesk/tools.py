"""
wealthdesk/tools.py
-------------------
LLM client setup.

Provided in full for Session 1 -- no changes needed here.
In later sessions this file will grow to include @tool functions
that let the agent query live databases.
"""
import os

from langchain_groq import ChatGroq

from .config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, CLASSIFICATION_MODEL_NAME, CLASSIFICATION_TEMPERATURE, CLASSIFICATION_MAX_TOKENS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

# LLM clients for response generation
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

# classification model is deterministic, so we set temperature to 0.0 
# can configure a separate model for classification if needed, but for now we use the same model with different temperature
classifer_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=CLASSIFICATION_MODEL_NAME,
    temperature=CLASSIFICATION_TEMPERATURE,  # deterministic classification
    max_tokens=CLASSIFICATION_MAX_TOKENS,
)

print("=" * 55)
print(f"Intent Classification is done by LangGraph LLM Model: {classifer_llm.model}, with temperature {classifer_llm.temperature}")
print(f"Responses Generation is done by LangGraph LLM Model: {llm.model}, with temperature {llm.temperature}")
print("=" * 55)
