# Stage 4 — RAG

**Evaluates:** retrieval relevance, faithfulness
**Suggested tool (from evaluation plan):** Ragas

## What's actually testable here

Mocking the vectorstore lets us fully verify the deterministic filtering
logic in `retrieve_docs()` and the faithfulness *guardrail* in `respond()`
(escalate rather than hallucinate when nothing relevant was retrieved).
Actual semantic relevance/faithfulness of real retrieved chunks needs a
judge model — that's the Ragas part, run live.

## Part A — Mocked retrieval filtering tests

```python
from unittest.mock import patch
from langchain_core.documents import Document
import wealthdesk.nodes as _nodes
from wealthdesk.nodes import retrieve_docs, respond
from wealthdesk.config import RETRIEVAL_SCORE_THRESHOLD

def test_filters_out_low_score_chunks():
    with patch.object(_nodes, "vectorstore") as mock_vs:
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Home loan needs PAN + Aadhaar.",
                      metadata={"source": "home_loan_guide.md"}), 0.82),
            (Document(page_content="Unrelated cricket score chunk.",
                      metadata={"source": "junk.md"}), 0.12),
        ]
        result = retrieve_docs({"customer_message": "What documents do I need for a home loan?"})
        assert len(result["retrieved_docs"]) == 1
        assert "home_loan_guide.md" in result["retrieved_docs"][0]

def test_all_chunks_below_threshold_returns_empty():
    with patch.object(_nodes, "vectorstore") as mock_vs:
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="x", metadata={"source": "a.md"}), RETRIEVAL_SCORE_THRESHOLD - 0.01),
        ]
        result = retrieve_docs({"customer_message": "irrelevant question"})
        assert result["retrieved_docs"] == []

def test_vectorstore_unavailable_returns_empty_not_crash():
    with patch.object(_nodes, "vectorstore", None), \
         patch.object(_nodes, "_init_vectorstore", lambda: None):
        result = retrieve_docs({"customer_message": "anything"})
        assert result["retrieved_docs"] == []
```

## Part B — Faithfulness guardrail: escalate on empty context

No relevant docs retrieved → escalate to a human, never let the LLM guess.

```python
def test_respond_escalates_when_no_docs_retrieved():
    state = {
        "customer_message": "What is BNB's crypto custody policy?",
        "history": [], "retrieved_docs": [],
    }
    with patch.object(_nodes, "llm_with_tools") as mock_llm:
        result = respond(state)
        mock_llm.invoke.assert_not_called()   # never even calls the LLM
        from wealthdesk.config import ESCALATE_RESPONSE
        assert result["response"] == ESCALATE_RESPONSE
```

## Part C — Live Ragas eval (manual run, needs a judge-model API key)

Sample golden set:

```python
RAG_GOLDEN_SET = [
    {"question": "What documents do I need for a home loan?",
     "ground_truth": "PAN card, Aadhaar, income proof, and property documents."},
    {"question": "What is the minimum tenure for a fixed deposit?",
     "ground_truth": "BNB offers FD tenures starting from 7 days up to 10 years."},
]
```

`s01/tests/live_eval_rag.py` (not run in CI):

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy

def run_live_eval():
    # For each row in RAG_GOLDEN_SET: run the real retrieve_docs() + respond(),
    # collect {question, contexts, answer, ground_truth}, then:
    rows = [...]  # populate from real graph runs
    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[context_precision, faithfulness, answer_relevancy])
    print(result)
```

Requires `ragas`, `datasets`, and a real embedding/vectorstore plus a judge
LLM (typically OpenAI, configurable in Ragas) — install and key setup is a
prerequisite before this can run.
