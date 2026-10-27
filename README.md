# Autonomous Bug Investigator

Give it a GitHub issue URL. It searches the repository's code, docs, past issues and commit history, works out the likely root cause, and returns a debugging report with cited evidence, a suggested fix and test cases.

A LangGraph workflow of four agents — **planner → investigator → fix → critic** — runs the investigation. When the critic judges an answer weak or ungrounded, it sends it back to the planner for another pass with specific feedback. Every run is recorded in MySQL, step by step.

**Stack:** Python · LangGraph · FastAPI · FAISS + BM25 · MySQL · GitHub API · React

---

## Why this isn't just a RAG demo

**Hybrid retrieval, because neither half works alone on code.** Dense FAISS vectors handle paraphrase — "crashes on empty input" finding a function that divides by `len()`. BM25 handles the exact identifiers and error strings that embeddings blur together. The two rankings are merged with Reciprocal Rank Fusion. The tokenizer splits `getUserById` into its parts, so searching for `user` finds it.

**Citations enforced in code, not requested in a prompt.** Asking a model to "only cite real evidence" is a suggestion. Here, evidence IDs the model cites but never retrieved are dropped before the critic ever sees them, and an analysis left with no valid citation can never be accepted, whatever verdict the model returns. The guardrail is deterministic Python, not a hopeful instruction.

**A critic loop that actually loops.** A `revise` verdict routes back to the planner carrying the critic's feedback *and* the previous queries, so the next pass searches differently instead of repeating itself. Bounded by `MAX_ITERATIONS`; exhaust the passes and the report ships flagged `low_confidence` rather than pretending to be sure.

**Reproducible runs.** Each investigation records the model, the prompt version and the exact commit SHA of the index it searched. Re-indexing creates a new index version rather than mutating the old one, so a report from last week still means what it meant.

**Built to be measured.** The eval harness replays closed issues whose real fix PR is known and scores file-level hit@1 / hit@3 / recall against a BM25-only baseline.

---

## Design docs

| | |
|---|---|
| [01 · Requirements & scope](docs/01-requirements-scope.md) | functional and non-functional requirements, what's in and out |
| [02 · Architecture](docs/02-architecture.md) | ingestion and investigation flows, the agent graph, failure handling |
| [03 · Data model](docs/03-data-model.md) | 13 tables, ER diagram, why the schema is shaped this way |
| [04 · Tech stack](docs/04-tech-stack.md) | every choice and the alternative it beat |

---

## Architecture

```
GitHub issue URL
      │
      ▼
┌─────────────────────────────────────────────┐
│  LangGraph workflow                         │
│                                             │
│   planner ──► investigator ──► fix ──► critic
│      ▲                                   │  │
│      └───────── revise ◄──────────────────  │
│                                    accept   │
└───────────────────────────────────┬─────────┘
                                    ▼
        Hybrid retrieval            report + fix + tests
     ┌──────────┬──────────┐              │
     │  FAISS   │  BM25    │              ▼
     │  (dense) │ (sparse) │           MySQL
     └─────┬────┴─────┬────┘      (every step, every
           └─── RRF ───┘           retrieval, every
                                   critic verdict)
```

Ingestion runs separately, once per repository: clone → chunk code, docs, issues and commits → embed → FAISS index + MySQL rows. Re-running is a no-op unless HEAD moved.

---

## Running it

```bash
cp .env.example .env     # fill in GITHUB_TOKEN and GOOGLE_API_KEY
docker compose up -d mysql

cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                                              # 107 tests, no API key or network needed

python -m app.cli index pallets/click
python -m app.cli run https://github.com/pallets/click/issues/3887
uvicorn app.main:app --reload                       # http://localhost:8000/docs
```

```bash
cd frontend && npm install && npm run dev           # http://localhost:5173
```

A free Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) is enough to run everything.

### CLI

| Command | What it does |
|---|---|
| `python -m app.cli initdb` | create the schema without Docker |
| `python -m app.cli index owner/repo` | clone and index a repository |
| `python -m app.cli index owner/repo --issue-limit 0 --commit-limit 0` | code and docs only — much faster first run |
| `python -m app.cli run <issue-url>` | investigate one issue, print the Markdown report |
| `python -m app.cli eval owner/repo --build-cases 15` | score the agent against the BM25 baseline |

### Configuration

| Variable | Notes |
|---|---|
| `LLM_PROVIDER` | `gemini` (default), `anthropic`, `openai`, `ollama` |
| `LLM_MODEL` | model names change; check your provider's current list |
| `EMBEDDING_MODEL` | a sentence-transformers name (best), `gemini:<model>` (no local torch), or `hashing` (offline, poor quality, used by tests) |
| `MAX_ITERATIONS` | critic passes before giving up, default 3 |
| `TOKEN_BUDGET` | hard stop per run |

---

## Evaluation

```
$ python -m app.cli eval pallets/click --build-cases 15

system             n    hit@1    hit@3   recall
-----------------------------------------------
agent              ?        ?        ?        ?
bm25_baseline      ?        ?        ?        ?
```

**These are not results.** That's the output format; the harness has not yet been run against real repositories at scale, so there is no measured number to report. The metrics and the case-selection logic are built and unit-tested — the run is the remaining work.

Eval cases are built only from closed issues linked to exactly one fix PR touching five or fewer non-test files, so the ground truth is unambiguous. The baseline matters as much as the agent: a localization score with nothing to compare against says nothing about whether the agentic layer earns its cost.

---

## Tests

```
backend/tests/
  test_units.py      JSON extraction, RRF, URL parsing, retry/backoff, token budgeting, ORM
  test_graph.py      critic routing and citation guardrails against a scripted LLM
  test_ingest.py     chunking rules, real git walking, index building, clone-race recovery
  test_retrieval.py  tokenization and hybrid search against a real FAISS index
  test_runner.py     a full run: real index, real retrieval, real persistence
  test_api.py        every endpoint, GitHub and the LLM stubbed
  test_eval.py       metrics and harness wiring
```

107 tests. Nothing hits the network or needs an API key. `conftest.py` builds a throwaway git repository containing a real bug — `average()` divides by `len()` with no empty-list guard — and indexes it for real, so retrieval and persistence are tested against genuine artifacts rather than mocks.

---

## Status

| Milestone | State |
|---|---|
| Requirements, architecture, data model | done |
| Ingestion + hybrid retrieval | done |
| Agent loop + citation guardrails | done |
| Runner, MySQL persistence, API, UI | done |
| Eval harness | built and unit-tested; not yet run at scale |

Verified end to end against `pallets/click`: 889 chunks indexed, planner and investigator producing grounded output on a real issue.

**Not built:** sandboxed execution of the suggested fix and tests, symbol-level (tree-sitter) chunking, SSE progress instead of polling, private-repo authentication.

---

## Things that only showed up by running it

Worth recording, because none of these appeared in tests:

- **A clone killed partway** leaves a directory `git clone` refuses to reuse forever. Two indexing jobs racing on one repository produced exactly that. Fixed with an atomic-`mkdir` lock and a validity check that removes and re-clones a broken checkout.
- **Batching embeddings by item count** rather than token count meant a single request could exceed a whole minute's quota — so every retry failed no matter how long it waited. Now batched by estimated tokens and paced against a sliding 60-second window.
- **Rate-limit responses carry the wait time.** Honoring the API's own `retryDelay` beats doubling a guess — but a multi-hour delay means a daily quota, not a burst, so it now fails loudly instead of sleeping until tomorrow.
- **Transient 503s are not failures.** A provider under load should be retried; a 400 should not.
- **Issue history is a bonus, not a requirement.** A GitHub hiccup used to throw away an otherwise complete code index.
