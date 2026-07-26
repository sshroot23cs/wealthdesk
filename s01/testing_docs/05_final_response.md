# Stage 5 — Final Response

**Evaluates:** correctness, groundedness, completeness
**Suggested tool (from evaluation plan):** DeepEval or LLM-as-a-judge

## What's actually testable here

Mocked tests can't judge semantic correctness of a real answer, but they
can enforce the **structural** rules baked into `SYSTEM_PROMPT` (word cap,
no markdown tables, mandatory sign-off) and verify **groundedness at the
plumbing level** — that real tool output is what actually reaches the final
LLM call, not something reconstructed from the question.

## Part A — Response format validator (structural rules)

```python
import pytest

def assert_format_compliant(text: str):
    assert len(text.split()) <= 160          # ~150-word cap from SYSTEM_PROMPT + slack
    assert "|" not in text.replace("WealthDesk | Bharat National Bank", "")
    assert "WealthDesk | Bharat National Bank" in text

def test_validator_accepts_compliant_sample():
    sample = ("The BNB home loan rate is 8.50% p.a.\n"
              "WealthDesk | Bharat National Bank")
    assert_format_compliant(sample)

def test_validator_rejects_markdown_table():
    bad = "| Rate | Value |\n|---|---|\nWealthDesk | Bharat National Bank"
    with pytest.raises(AssertionError):
        assert_format_compliant(bad)
```

This validator is a reusable tripwire — apply it to *real* model output in
the live eval below to catch prompt-drift regressions (e.g. a model upgrade
that starts ignoring the "no markdown" rule).

## Part B — Groundedness plumbing test

The tool's actual output must be the thing the final LLM call sees.

```python
from unittest.mock import patch
from langchain_core.messages import ToolMessage
import wealthdesk.nodes as _nodes
from wealthdesk.nodes import respond

def test_tool_output_reaches_final_llm_call_as_tool_message():
    state = {
        "customer_message": "What is the home loan rate?",
        "history": [], "retrieved_docs": ["[doc] ..."],
    }
    with patch.object(_nodes, "llm_with_tools") as mock_llm, \
         patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.50% p.a., tenure 5-30 years"):
        mock_llm.invoke.side_effect = [
            _tool_call_result([{"name": "query_rates", "args": {"product_type": "loan"}, "id": "t1"}]),
            _final_result("The home loan rate is 8.50% p.a. WealthDesk | Bharat National Bank"),
        ]
        respond(state)
        final_call_messages = mock_llm.invoke.call_args_list[-1].args[0]
        tool_messages = [m for m in final_call_messages if isinstance(m, ToolMessage)]
        assert any("8.50%" in m.content for m in tool_messages)
```

(`_tool_call_result` / `_final_result` helpers are the same as in
[03_tool_calling.md](03_tool_calling.md).)

## Part C — Live correctness/groundedness/completeness eval (DeepEval)

`s01/tests/live_eval_response.py` (not run in CI, needs real `GROQ_API_KEY`
and `deepeval`):

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def run_live_eval():
    # Run the real graph for a question, capture:
    #   input               = customer question
    #   actual_output       = agent's final response
    #   retrieval_context   = state["retrieved_docs"]
    # then score:
    test_case = LLMTestCase(
        input="What documents do I need for a home loan?",
        actual_output="...",           # from a real graph.invoke()
        retrieval_context=["..."],      # from state["retrieved_docs"]
    )
    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = AnswerRelevancyMetric(threshold=0.7)
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    print(faithfulness.score, relevancy.score)
```

## Not covered by mocked tests

Whether a real answer is factually correct (e.g. quotes the actual current
rate rather than a stale/hallucinated one) and complete (addresses every
part of a multi-part question) — both require either a live judge model or
manual review against the golden set.
