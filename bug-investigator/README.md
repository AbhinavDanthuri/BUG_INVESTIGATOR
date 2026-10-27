# Autonomous Bug Investigator

Give it a GitHub issue URL. It searches the repository's code, docs, past issues and commits, works out the likely root cause, and returns a debugging report with a suggested fix and test cases. A LangGraph workflow of **planner → investigator → fix → critic** runs the investigation, and the critic sends weak answers back for another pass. Every run is stored in MySQL.

**Stack:** Python · LangGraph · FastAPI · FAISS + BM25 · MySQL · GitHub API · React

## What makes it more than a RAG demo

- **Hybrid retrieval.** Dense FAISS vectors catch paraphrase ("crashes on empty input"); BM25 catches exact identifiers and error strings. The two rankings are merged with Reciprocal Rank Fusion, because neither alone works on code.
- **Citations are enforced in code, not asked for in a prompt.** Evidence ids the model cites but never retrieved are dropped before the critic sees them, and an analysis left with no valid citation can never be accepted, whatever verdict the model returns.
- **The critic loop is real.** A `revise` verdict routes back to the planner with feedback and the previous queries, up to `MAX_ITERATIONS`. Run out of passes and the report ships flagged `low_confidence` rather than pretending.
- **Reproducible runs.** Each investigation records the model, prompt version and the exact indexed commit SHA. Re-indexing creates a new index version instead of mutating the old one, so a past run still means what it meant.
- **It is measured.** The eval harness replays closed issues whose real fix PR is known and scores file-level hit@1 / hit@3 / recall against a BM25-only baseline.

## Design docs
1. [Requirements & scope](docs/01-requirements-scope.md)
2. [Architecture](docs/02-architecture.md)
3. [Data model](docs/03-data-model.md)
4. [Tech stack](docs/04-tech-stack.md)

## Run it

```bash
cp .env.example .env     # add GITHUB_TOKEN and GOOGLE_API_KEY
docker compose up -d mysql          # schema loads from backend/sql/schema.sql

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                              # 78 tests, no API key and no network needed

python -m app.cli index psf/requests
python -m app.cli run https://github.com/psf/requests/issues/6710
uvicorn app.main:app --reload       # http://localhost:8000/docs
```

```bash
cd frontend && npm install && npm run dev    # http://localhost:5173
```

A free Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) is enough; set `LLM_MODEL` to whatever Flash model AI Studio currently lists. Rate-limit errors are retried with exponential backoff.

## CLI

| Command | What it does |
|---------|--------------|
| `python -m app.cli initdb` | Create the schema (alternative to docker-compose) |
| `python -m app.cli index owner/repo [--force]` | Clone and index a repo |
| `python -m app.cli run <issue-url>` | Investigate one issue, print the Markdown report |
| `python -m app.cli eval owner/repo --build-cases 15` | Discover eval cases, then score agent vs BM25 baseline |

## Evaluation

```
$ python -m app.cli eval psf/requests --build-cases 15

system             n    hit@1    hit@3   recall
-----------------------------------------------
agent              ?        ?        ?        ?
bm25_baseline      ?        ?        ?        ?
```

**No results yet — that is the output format, not measured numbers.** The harness and the metrics are built and unit-tested, but it has not been run against real repositories, which needs a GitHub token and an LLM key. Until it has been, there is no honest figure to quote for this project. Run it first, then put the number on a CV.

## Embeddings

Default is `BAAI/bge-small-en-v1.5` via sentence-transformers, downloaded once on first use. Setting `EMBEDDING_MODEL=hashing` selects a dependency-free deterministic embedder instead: it needs no download and runs anywhere, which is what the test suite uses. Its retrieval quality is poor, so it is for tests and offline smoke runs, not for real investigations.

## Tests

```
backend/tests/
  test_units.py      JSON extraction, RRF, URL parsing, LLM retry/backoff, ORM round-trip
  test_graph.py      critic routing and citation guardrails against a fake LLM
  test_ingest.py     chunking rules, real git walking, index building, clone-race recovery
  test_retrieval.py  tokenization and hybrid search against a real FAISS index
  test_runner.py     full run end to end: real index, real retrieval, real persistence
  test_api.py        every endpoint, with GitHub and the LLM stubbed
  test_eval.py       metrics and harness wiring
```

Nothing in the suite hits the network or needs an API key. `conftest.py` builds a real throwaway git repo containing a real bug (`average()` divides by `len()` with no empty-list guard) and indexes it for real.

## Status

| Milestone | State |
|-----------|-------|
| M0 design + scaffold | done |
| M1 ingestion + hybrid retrieval | done |
| M2 agent loop + guardrails | done |
| M3 runner + persistence + API + UI | done |
| M4 eval harness + baseline | built, not yet run on real repos |

Not done yet: running the suggested fix or tests in a sandbox, symbol-level (tree-sitter) chunking, SSE progress instead of polling, and private-repo auth.
