"""
wealthdesk/tools.py
-------------------
LLM client setup.

Provided in full for Session 1 -- no changes needed here.
In later sessions this file will grow to include @tool functions
that let the agent query live databases.
"""
import asyncio
import os
import sqlite3
import sys

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import MAX_TOKENS, MCP_SERVER_PATH, MODEL_NAME, TEMPERATURE, CLASSIFICATION_MODEL_NAME, CLASSIFICATION_TEMPERATURE, CLASSIFICATION_MAX_TOKENS

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
classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=CLASSIFICATION_MODEL_NAME,
    temperature=CLASSIFICATION_TEMPERATURE,  # deterministic classification
    max_tokens=CLASSIFICATION_MAX_TOKENS,
)

_BOX_WIDTH = 100
_BOX_INNER_WIDTH = _BOX_WIDTH - 4  # "X " prefix + " X" suffix
_box_lines = [
    "* WealthDesk Agent",
    "",
    f"Intent Classification Model: {classifier_llm.model}, with temperature {classifier_llm.temperature}",
    f"Responses Generation Model: {llm.model}, with temperature {llm.temperature}",
    "",
    "Built By: Sushrut Hole",
]


def _print_box(lines, inner_width):
    # Plain ASCII only: this runs at import time, and some hosts (e.g. langgraph
    # dev's worker threads on Windows) redirect stdout through a non-UTF-8 codec,
    # which raises UnicodeEncodeError on box-drawing characters and aborts the import.
    print("+" + "-" * (inner_width + 2) + "+")
    for line in lines:
        if len(line) > inner_width:
            line = line[: inner_width - 3] + "..."
        print(f"| {line:<{inner_width}} |")
    print("+" + "-" * (inner_width + 2) + "+")


_print_box(_box_lines, _BOX_INNER_WIDTH)

_mcp_client = MultiServerMCPClient({
      "wealthdesk": {
          "transport": "stdio",
          "command": sys.executable,
          "args": [str(MCP_SERVER_PATH)],
      }
  })

mcp_tools      = asyncio.run(_mcp_client.get_tools())
_tool_registry = {t.name: t for t in mcp_tools}
llm_with_tools = llm.bind_tools(mcp_tools)

def _extract_text(result) -> str:
    """MCP tool results come back as a list of content blocks, e.g.
    [{"type": "text", "text": "...", "id": "..."}]. Join the text blocks
    into the plain string the rest of the agent code expects. Provided --
    no changes needed."""
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)
 
 
def _run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in _tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        # Tools loaded via langchain-mcp-adapters are async-only (they
        # negotiate a fresh MCP session per call) -- bridge to the
        # synchronous LangGraph node with asyncio.run().
        result = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
        return _extract_text(result)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
