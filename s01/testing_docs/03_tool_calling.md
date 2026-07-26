# Stage 3 — Tool Calling

**Evaluates:** correct tool, correct arguments, correct sequence
**Suggested tool (from evaluation plan):** LangSmith traces, custom assertions

## Helpers

```python
from unittest.mock import MagicMock, patch
import wealthdesk.nodes as _nodes
import wealthdesk.tools as _tools
from wealthdesk.nodes import respond

def _tool_call_result(calls):
    """Build a mock LLM result carrying tool_calls, mimicking langchain's shape."""
    r = MagicMock()
    r.tool_calls = calls
    r.invalid_tool_calls = []
    r.content = ""
    return r

def _final_result(text):
    r = MagicMock()
    r.tool_calls = []
    r.invalid_tool_calls = []
    r.content = text
    return r
```

## Sample prompts

- `"What is the home loan interest rate?"` → expects `query_rates(product_type="loan")`
- `"What are your FD rates for senior citizens?"` → expects `query_rates(product_type="fd")`
- `"Where is your nearest branch in Bengaluru?"` → expects `query_branch(city="Bengaluru")`
- `"What are the FD rates, and where is your Pune branch?"` → expects a
  **sequence**: `query_rates` then `query_branch` (gpt-oss-20b calls one tool
  per round, so this requires two LLM round-trips)

## Test cases

### 3a. Correct tool + correct arguments

```python
def test_query_rates_called_with_correct_args():
    state = {
        "customer_message": "What is the home loan interest rate?",
        "history": [], "retrieved_docs": ["[home_loan_guide.md] ..."],
    }
    with patch.object(_nodes, "llm_with_tools") as mock_llm, \
         patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.50% p.a.") as mock_run:
        mock_llm.invoke.side_effect = [
            _tool_call_result([{"name": "query_rates", "args": {"product_type": "loan"}, "id": "t1"}]),
            _final_result("The BNB home loan rate is 8.50% p.a. WealthDesk | Bharat National Bank"),
        ]
        respond(state)
        mock_run.assert_called_once_with("query_rates", {"product_type": "loan"})
```

### 3b. Correct sequence across two round-trips

```python
def test_sequential_tool_calls_rates_then_branch():
    state = {
        "customer_message": "What are the FD rates, and where is your Pune branch?",
        "history": [], "retrieved_docs": ["[doc] ..."],
    }
    with patch.object(_nodes, "llm_with_tools") as mock_llm, \
         patch.object(_nodes, "_run_tool") as mock_run:
        mock_run.side_effect = ["FD rates: ...", "Pune branch: ..."]
        mock_llm.invoke.side_effect = [
            _tool_call_result([{"name": "query_rates", "args": {"product_type": "fd"}, "id": "t1"}]),
            _tool_call_result([{"name": "query_branch", "args": {"city": "Pune"}, "id": "t2"}]),
            _final_result("Here are the FD rates and the Pune branch. WealthDesk | Bharat National Bank"),
        ]
        respond(state)
        called_names = [c.args[0] for c in mock_run.call_args_list]
        assert called_names == ["query_rates", "query_branch"]
```

### 3c. Loop guard — must not hang if the LLM keeps requesting tools

```python
def test_tool_loop_stops_at_max_rounds():
    state = {"customer_message": "rates please", "history": [], "retrieved_docs": ["[doc] ..."]}
    with patch.object(_nodes, "llm_with_tools") as mock_llm, \
         patch.object(_nodes, "_run_tool", return_value="..."):
        mock_llm.invoke.return_value = _tool_call_result(
            [{"name": "query_rates", "args": {}, "id": "loop"}]
        )
        respond(state)
        # 1 initial call + 5 rounds (max_tool_rounds) = 6 invocations total
        assert mock_llm.invoke.call_count == 6
```

### 3d. Unknown tool name handled gracefully

```python
def test_run_tool_unknown_name_does_not_raise():
    result = _tools._run_tool("delete_account", {})
    assert result == "Unknown tool: delete_account"
```

### 3e. Underlying tool exception is caught and reported, not raised

```python
def test_run_tool_catches_underlying_exception():
    with patch.object(_tools, "_execute_query", side_effect=RuntimeError("db locked")):
        wrapped = _tools._run_tool("query_rates", {"product_type": "loan"})
        assert "Tool error" in wrapped
```

## Not covered by mocked tests

Whether the *real* LLM chooses the right tool for a given prompt (as opposed
to whichever tool the test hands it) — that requires either live calls or
inspecting real LangSmith traces for tool-selection accuracy across a
prompt set.
