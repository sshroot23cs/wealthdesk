# WealthDesk Evaluation Test Cases

Test-case documentation for the `s01/starter/wealthdesk` agent, organized by
evaluation stage. These are fully-mocked test specs (no live `GROQ_API_KEY`
or extra packages required) matching the conventions already used in
`s01/tests/test_s01.py` and `s01/tests/conftest.py` (dummy API keys,
`patch.object(_nodes, ...)` / `patch.object(_tools, ...)` for mocking LLM
and DB calls).

| # | Stage | What it evaluates | File |
|---|-------|--------------------|------|
| 1 | Guardrails | Prompt injection, out-of-scope detection, SQL injection defense | [01_guardrails.md](01_guardrails.md) |
| 2 | Intent Classifier | Intent accuracy (SIMPLE / COMPLEX / OUT_OF_SCOPE) | [02_intent_classifier.md](02_intent_classifier.md) |
| 3 | Tool Calling | Correct tool, arguments, and call sequence | [03_tool_calling.md](03_tool_calling.md) |
| 4 | RAG | Retrieval relevance, faithfulness (no-context escalation) | [04_rag.md](04_rag.md) |
| 5 | Final Response | Correctness, groundedness, completeness/format | [05_final_response.md](05_final_response.md) |
| 6 | Production | Latency, cost bounds, failure resilience, observability | [06_production.md](06_production.md) |

## Important caveat

Rows **2 (Intent Classifier)**, **4 (RAG)**, and **5 (Final Response)**
fundamentally require a *real* LLM/embedding call to measure what they claim
to measure — classification accuracy, retrieval faithfulness, and answer
correctness can't be verified against a mocked model that just returns
whatever canned value the test tells it to.

For those three, each doc includes two parts:

1. **Mocked unit tests** — verify the deterministic code paths those stages
   depend on (parsing/fallback logic, score-threshold filtering, message
   plumbing, format rules). These are safe for CI.
2. **Live eval sketch** — a golden dataset + a script meant to run against
   the real `classifier_llm` / `llm` / vectorstore (using `sklearn` for
   confusion matrix/F1, `ragas` for retrieval metrics, `deepeval` for
   answer-quality metrics). These are **not** meant for CI — run manually
   with real API keys when you want an actual accuracy/quality number.
   They follow the same intent as the existing `s05/tests/live_eval.py`.

## Target module

All tests target `s01/starter/wealthdesk` (the fully implemented version —
tool calling, RAG, 3-way classification all present). `s05/starter/wealthdesk`
currently has unfilled TODOs (`query_rates`/`query_branch` are `pass`,
`llm_with_tools` not bound), so these specific test cases will fail there
until that exercise is completed.

## Status

These files are **documentation/specs**, not committed `.py` test files.
Turn any of them into runnable pytest files under `s01/tests/` on request.
