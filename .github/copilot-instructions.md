## Copilot / AI Agent Instructions for WealthDesk

Purpose: give an AI coding agent the minimal, actionable knowledge to be productive in this repo.

- **Big picture**: WealthDesk is a teaching repo that builds a multi-session banking assistant.
  - Data layer: `data/bnb_data.db` (SQLite) seeded by [data/seed.py](data/seed.py). Rates and numeric truth live here.
  - Document retrieval: ChromaDB vector store in `data/vectorstore/`, created by [data/ingest.py](data/ingest.py) from `data/documents/`.
  - Session code: each session under `s01/`, `s02/` …; starter code you edit is in `s01/starter/` and reference solutions in `s01/solution/`.
  - LLM: Session 1 uses Groq via `langchain_groq.ChatGroq` (see [s01/starter/main.py](s01/starter/main.py)).

- **Primary constraints / design decisions** (do NOT change unless instructed):
  - Rates and structured numbers are authoritative in SQLite (`data/bnb_data.db`) — do not duplicate them into documents.
  - `data/ingest.py` deletes and rebuilds the vector store on every run (idempotent behavior).
  - Tests mock the LLM; tests must remain runnable offline (`pytest s01/tests/ -v`).

- **Key developer workflows & commands**
  - Install deps: `pip install -r requirements.txt` (run from repo root).
  - Seed DB then ingest docs (one-time per workspace):
    - `python data/seed.py`
    - `python data/ingest.py`
  - Quick setup verification: `python test_setup.py`
  - Run session 1 agent (after filling TODOs): `python s01/starter/main.py` or use solution at `s01/solution/main.py`.
  - Run tests: `pytest s01/tests/ -v` (tests do not require a live Groq key).

- **Patterns & conventions to follow when editing code**
  - Always run scripts from the repository root (README enforces this).
  - Environment: call `load_dotenv()` before any `os.getenv()` use (see TODO in [s01/starter/main.py](s01/starter/main.py)).
  - System prompt: tests expect specific content (mentions `WealthDesk`, `Bharat National Bank` / `BNB`, decline out-of-scope rule and include rates). See `s01/tests/test_s01.py` for checks.
  - TypedDicts: session TypedDicts must declare fields explicitly (tests inspect `WealthDeskState.__annotations__`).
  - LLM node contract: `respond()` must build messages [SystemMessage, HumanMessage], call `llm.invoke(messages)`, return a dict with changed keys (eg `{"response": ...}`), and handle exceptions with a safe fallback message (no internal error strings).

- **Integration points & external deps**
  - Groq LLM: configured via `GROQ_API_KEY` in `.env` and used by `langchain_groq.ChatGroq`.
  - Embeddings: local HuggingFace model `all-MiniLM-L6-v2` is used by ingest (cached to `~/.cache/huggingface/`).
  - ChromaDB: persisted to `data/vectorstore/` (Windows-safe path handling already implemented).
  - LangGraph: session graphs are built with `StateGraph` and nodes return partial-state dicts.

- **Where to make changes for common tasks**
  - Update rates or product rows: modify `data/seed.py` and re-run `python data/seed.py` (do NOT hardcode rates elsewhere).
  - Add or update policy text: edit `data/documents/*.md` then re-run `python data/ingest.py`.
  - Implement session code TODOs: edit `s01/starter/main.py` (follow tests for expected behavior).

If anything here is unclear or you want more examples (for example, exact prompt wording or test-driven fix suggestions), tell me which section to expand.  