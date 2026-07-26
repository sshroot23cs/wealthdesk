# Stage 6 — Production

**Evaluates:** latency, cost, user feedback, failures
**Suggested tool (from evaluation plan):** LangSmith + observability dashboards

## What's actually testable here

Genuine latency/cost/user-feedback monitoring needs LangSmith traces against
real traffic — a mocked unit test can't honestly claim to measure that.
What mocking *can* cover: failure-resilience (the graph never crashes the
process, even when a dependency fails) and cost-bounding configuration
(token ceilings actually set as intended).

## Part A — Failure resilience

```python
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
import wealthdesk.nodes as _nodes
from wealthdesk.agent import build_graph

def test_tool_exception_does_not_crash_graph():
    graph = build_graph()
    state = {
        "customer_message": "What is the home loan rate?", "response": "",
        "history": [], "retrieved_docs": [],
    }
    with patch.object(_nodes, "classifier_llm") as mock_classifier, \
         patch.object(_nodes, "vectorstore") as mock_vs, \
         patch.object(_nodes, "llm_with_tools") as mock_llm:
        mock_classifier.invoke.return_value = MagicMock(content="SIMPLE")
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="rates info", metadata={"source": "rates.md"}), 0.9),
        ]
        mock_llm.invoke.side_effect = Exception("Groq 500")
        result = graph.invoke(state)
        assert "response" in result
        assert "temporarily unavailable" in result["response"].lower()
```

## Part B — Cost bounds (proxy checks)

```python
def test_classifier_max_tokens_is_small():
    from wealthdesk.config import CLASSIFICATION_MAX_TOKENS
    assert CLASSIFICATION_MAX_TOKENS <= 10   # single-word label, not a paragraph

def test_response_max_tokens_is_bounded():
    from wealthdesk.config import MAX_TOKENS
    assert 100 < MAX_TOKENS <= 600
```

## Part C — Latency regression tripwire (not a real SLA test)

A cheap guard against an accidental infinite loop or blocking call being
introduced into the mocked code path — not a measurement of real production
latency.

```python
import time

def test_mocked_full_graph_invoke_completes_quickly():
    graph = build_graph()
    state = {
        "customer_message": "What is the FD rate?", "response": "",
        "history": [], "retrieved_docs": [],
    }
    with patch.object(_nodes, "classifier_llm") as mock_classifier, \
         patch.object(_nodes, "vectorstore") as mock_vs, \
         patch.object(_nodes, "llm_with_tools") as mock_llm:
        mock_classifier.invoke.return_value = MagicMock(content="SIMPLE")
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="FD info", metadata={"source": "fd.md"}), 0.9),
        ]
        mock_llm.invoke.return_value = _final_result(
            "FD rate is 6.80%. WealthDesk | Bharat National Bank"
        )
        start = time.monotonic()
        graph.invoke(state)
        assert time.monotonic() - start < 1.0
```

(`_final_result` helper is the same as in
[03_tool_calling.md](03_tool_calling.md).)

## Not covered by mocked tests — and what to do instead

Real latency, cost, failure-rate, and user-feedback tracking require
production traffic, not pytest. The actionable path is:

1. `LANGSMITH_TRACING=true` is already wired in `agent.py`'s `run()` —
   turn it on for real sessions.
2. Build a LangSmith dashboard off the traced project for:
   - p50/p95 latency per node (`classify`, `retrieve_docs`, `respond`)
   - token usage / cost per session
   - tool-call error rate
   - escalation rate (how often `ESCALATE_RESPONSE` / `DECLINE_RESPONSE` fire)
3. Capture user feedback (thumbs up/down, or explicit corrections) tagged to
   the LangSmith run ID so it can be joined back to specific traces.

This is observability infrastructure, not something a test suite can stand in for.
