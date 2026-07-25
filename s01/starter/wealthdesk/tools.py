"""
wealthdesk/tools.py
-------------------
LLM client setup.

Provided in full for Session 1 -- no changes needed here.
In later sessions this file will grow to include @tool functions
that let the agent query live databases.
"""
import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import DB_PATH, MAX_TOKENS, MODEL_NAME, TEMPERATURE, CLASSIFICATION_MODEL_NAME, CLASSIFICATION_TEMPERATURE, CLASSIFICATION_MAX_TOKENS

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


def _get_db_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection to the given database path.

    check_same_thread=False: SQLite normally raises ProgrammingError if a connection
    is used from a thread other than the one that created it. LangGraph runs tool calls
    in a thread-pool executor, so the tool may execute on a different thread than the
    one that opened the connection. This flag disables that check.
    """
    try:
        return sqlite3.connect(str(db_path), check_same_thread=False)
    except Exception as e:
        raise RuntimeError(f"Could not connect to database at {db_path}: {e}")


def _execute_query(sql: str, params: tuple = ()) -> list:
    """Open a connection, run a single query, and return all rows.

    Single choke point for every SQL query the tools issue: owns the connection's
    open/close lifecycle and turns any DB-layer failure (locked file, malformed SQL,
    corrupt db, etc.) into a RuntimeError with context, instead of letting a raw
    sqlite3 exception surface to the caller. Callers (query_rates, query_branch) don't
    need their own try/except/close around each query -- _run_tool already catches
    exceptions raised by tools and reports them back as a "Tool error" string.
    """
    conn = None
    try:
        conn = _get_db_connection(DB_PATH)
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database query failed: {e}") from e
    finally:
        if conn is not None:
            conn.close()


@tool
def query_rates(product_type: str = "all") -> str:
    """Fetch current BNB interest rates from the database.

    Args:
        product_type: Which rates to return. Options:
            "loan" -- all loan products (home, personal, car, education, gold)
            "fd"   -- all fixed deposit products
            "all"  -- both loans and FDs (default)

    Returns formatted rate information as a plain-text string.
    """
    lines = []

    if product_type in ("loan", "all"):
        rows = _execute_query(
            "SELECT name, interest_rate, tenure_min_years, tenure_max_years "
            "FROM loan_products ORDER BY interest_rate"
        )
        for name, rate, min_y, max_y in rows:
            lines.append(f"{name}: {rate:.2f}% p.a., tenure {min_y}-{max_y} years")

    if product_type in ("fd", "all"):
        rows = _execute_query(
            "SELECT tenure_label, interest_rate, senior_rate "
            "FROM fd_products ORDER BY tenure_months"
        )
        for label, rate, senior in rows:
            lines.append(
                f"FD {label}: {rate:.2f}% p.a. "
                f"(senior citizens: {rate + senior:.2f}%, extra +{senior:.2f}%)"
            )

    return "\n".join(lines) if lines else "No rate data found."


@tool
def query_branch(city: str = "all") -> str:
    """Fetch BNB branch locations from the database.

    Args:
        city: Filter branches by city name. Use "all" for every branch.

    Returns branch names, addresses, IFSC codes, and phone numbers.
    """
    if city.lower() == "all":

        rows = _execute_query(
            "SELECT name, city, address, ifsc, phone FROM branches ORDER BY city, name"
        )

    else:

        rows = _execute_query(
            "SELECT name, city, address, ifsc, phone "
            "FROM branches WHERE city LIKE ? ORDER BY name",
            (f"%{city}%",),
        )

        if not rows:
            # Neighbourhood names (e.g. "Andheri West", "Koramangala") are stored
            # in the branch name, not the city column — retry by name.
            rows = _execute_query(
                "SELECT name, city, address, ifsc, phone "
                "FROM branches WHERE name LIKE ? ORDER BY name",
                (f"%{city}%",),
            )

    if not rows:
        return f"No BNB branches found for city: '{city}'."
 
    parts = []
    for name, city_, address, ifsc, phone in rows:
        parts.append(
            f"{name} ({city_})\n"
            f"  Address: {address}\n"
            f"  IFSC: {ifsc}  |  Phone: {phone}"
        )
    return "\n\n".join(parts)
 


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- Bind tools to the LLM
# ---------------------------------------------------------------------------
# Create llm_with_tools by binding both tools to llm:
#   llm_with_tools = llm.bind_tools([query_rates, query_branch])
#
# bind_tools() reads each function's type hints and docstring, converts them to a
# JSON schema, and includes that schema in every request. The LLM was fine-tuned
# on examples with this schema format, so it knows to output a structured tool call
# instead of guessing the answer from memory.
# llm_with_tools is used for the FIRST call in respond(). The second call
# (after tools have run) uses plain llm.
# ---------------------------------------------------------------------------
# TODO: add llm_with_tools = llm.bind_tools([query_rates, query_branch])

AVAILABLE_TOOLS = [query_rates, query_branch]

def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call by name. Provided -- no changes needed."""
    _registry = {t.name: t for t in AVAILABLE_TOOLS}
    if tool_name not in _registry:
        return f"Unknown tool: {tool_name}"
    try:
        return _registry[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"

llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)