# Autonomous Bug Investigator

Give it a GitHub issue URL. It searches the repository's code, docs, past issues and commit history, works out the likely root cause, and returns a debugging report with cited evidence, a suggested fix and test cases.

A LangGraph workflow of four agents — **planner → investigator → fix → critic** — runs the investigation. When the critic judges an answer weak or ungrounded, it sends it back to the planner for another pass with specific feedback. Every run is recorded in MySQL, step by step.

**Stack:** Python · LangGraph · FastAPI · FAISS + BM25 · MySQL · GitHub API · React

---

## Why this isn't just a RAG demo

**Hybrid retrieval, because neither half works alone on code.** Dense FAISS vectors handle paraphrase — "crashes on empty input" finding a function that divides by `len()`. BM25 handles the exact identifiers and error strings that embeddings blur together. The two rankings are merged with Reciprocal Rank Fusion. The tokenizer splits `getUserById` into its parts, so searching for `user` finds it.

**It can conclude "this isn't a bug."** Most issue-triage tools are structurally incapable of that: every prompt assumes a defect exists, so they always manufacture a root cause and a patch. The investigator here returns `is_defect: yes | no | unclear`, and the graph routes around the fix agent entirely when the answer isn't `yes`. A report that says "this works as documented, here is the mechanism, a maintainer should decide" is a valid and useful output.

**It looks things up, not only searches.** A bug report usually says where it hurts — `core.py:1391`, a GitHub permalink, "line 1391". Similarity search is the wrong instrument for an address already written down. Those locations are extracted from the issue and fetched directly, and the planner can request a lookup when the critic asks for a specific definition. This came out of a real run: the critic rejected an analysis three times asking for `core.py:1391`, and the retriever — which could only search — never delivered it.

Three rules, each written after watching it fail:

- **A lookup without a line number returns nothing.** The first version fetched the top of the file instead — import statements, which say nothing about any bug.
- **Lookups are capped at half the evidence budget.** Those useless header chunks had filled every slot, starved search completely, and turned a correct verdict into "unclear".
- **Paths are matched from the right, by component.** Issues are written by hand and often carry a repo-name prefix: pallets/click#3832 points at `click/src/click/core.py` while the index holds `src/click/core.py`. Matching one direction only, every such reference silently resolved to nothing — and the critic kept asking for evidence that could never arrive.

**Citations enforced in code, not requested in a prompt.** Asking a model to "only cite real evidence" is a suggestion. Here, evidence IDs the model cites but never retrieved are dropped before the critic ever sees them, and an analysis left with no valid citation can never be accepted, whatever verdict the model returns. A patch produced for something judged not to be a defect is likewise rejected deterministically. The guardrails are Python, not hopeful instructions.

**A critic loop that actually loops.** A `revise` verdict routes back to the planner carrying the critic's feedback *and* the previous queries, so the next pass searches differently instead of repeating itself. Bounded by `MAX_ITERATIONS`; exhaust the passes and the report ships flagged `low_confidence` rather than pretending to be sure.

**Reproducible runs.** Each investigation records the model, the prompt version and the exact commit SHA of the index it searched. Re-indexing creates a new index version rather than mutating the old one, so a report from last week still means what it meant.

**Measured on both halves of the job.** The eval replays closed issues of two kinds: those fixed by a known PR, scored on file-level hit@1 / hit@3 / recall against a BM25-only baseline; and those closed as *not planned*, scored on whether the tool correctly declined to patch them. The second kind exists because an eval built only from fixed issues cannot see the worst failure mode — inventing a defect in working code — since such issues are excluded by construction.

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
pytest                                              # 186 tests, no API key or network needed

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

system             n  loc   hit@1   hit@3  recall  verdict
----------------------------------------------------------
agent              ?    ?       ?       ?       ?        ?
bm25_baseline      ?    ?       ?       ?       ?      n/a
```

**These are not results.** That's the output format; the harness has not yet been run at scale, so there is no measured number to report. The metrics and case-selection logic are built and unit-tested — the run is the remaining work.

- **`loc`** — how many cases had a known fix, which are the only ones localization is scored on.
- **`hit@1` / `hit@3` / `recall`** — did the predicted files match the ones the real fix touched.
- **`verdict`** — share of cases where the tool correctly judged defect vs not-a-defect. The baseline has no notion of intent, so it is `n/a` rather than zero.

Positive cases come from issues closed by exactly one merged PR in the same repository that says it fixes the issue, touching five or fewer code files (tests and docs don't count, and a docs-only fix is skipped), so the ground truth is unambiguous. Negative cases come from issues closed as *not planned* with no fix, minus closes made for process reasons (labels like `rejected`, `duplicate`, `stale`). On a negative case, both `no` and `unclear` count as correct — declining to patch is the behaviour that matters.

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

186 tests. Nothing hits the network or needs an API key. `conftest.py` builds a throwaway git repository containing a real bug — `average()` divides by `len()` with no empty-list guard — and indexes it for real, so retrieval and persistence are tested against genuine artifacts rather than mocks.

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

## The failure that changed the design

The first complete run, against [pallets/click#3887](https://github.com/pallets/click/issues/3887), produced a confident report: a specific root cause in `src/click/termui.py`, three cited evidence chunks with real line numbers, a patch, and two tests. The critic accepted it on the first pass at 78% confidence.

It was wrong. The issue had been closed as *not planned* — the maintainer's position was that `style()` is not meant to override pre-styled text, and that callers should strip codes themselves. The patch the tool proposed (calling `unstyle()` inside `style()`) was close to the change that had been explicitly rejected, and would have silently discarded ANSI sequences a caller had put there deliberately.

The mechanism it described was plausibly correct. The *frame* was wrong, and nothing in the system could catch that:

- Every prompt assumed a defect existed, so the tool could only ever produce one.
- The critic scored groundedness, specificity and consistency. All three passed — the claims genuinely were grounded in real code. None of them asked whether this was a defect at all.
- The eval would have reported success, because it only sampled issues closed by a fix PR. Won't-fix issues were excluded by construction, so the one failure mode that matters most was invisible to the metric.

Three changes followed: the `is_defect` verdict, routing that skips the fix agent entirely when the answer isn't `yes`, and negative eval cases built from *not planned* closures. The guardrail that catches ungrounded answers is a different thing from one that catches confident, well-cited, wrong-framed answers — and the second is harder and more important.

## Things that only showed up by running it

Worth recording, because none of these appeared in tests:

- **A clone killed partway** leaves a directory `git clone` refuses to reuse forever. Two indexing jobs racing on one repository produced exactly that. Fixed with an atomic-`mkdir` lock and a validity check that removes and re-clones a broken checkout.
- **Batching embeddings by item count** rather than token count meant a single request could exceed a whole minute's quota — so every retry failed no matter how long it waited. Now batched by estimated tokens and paced against a sliding 60-second window.
- **Rate-limit responses carry the wait time.** Honoring the API's own `retryDelay` beats doubling a guess — but a multi-hour delay means a daily quota, not a burst, so it now fails loudly instead of sleeping until tomorrow.
- **Transient 503s are not failures.** A provider under load should be retried; a 400 should not.
- **The first eval's ground truth was wrong in three ways.** Reading the individual cases showed a "fix PR" that was a different repository's PR #1, a docs-only fix scored as a missed bug, and a "won't fix" that was really a rejected AI-generated report. Each case-builder rule above exists because of one of those.
- **The fix agent ignored the diff format however it was asked.** Even with a worked example in the prompt it kept writing `*** Begin Patch` blocks with no line numbers. Prompting didn't fix it, so the conversion is code: the patch's context lines are located in the evidence the model was shown, which yields real line numbers and a valid unified diff. When a patch can't be placed, the report says so instead of presenting it as applicable.
- **Issue history is a bonus, not a requirement.** A GitHub hiccup used to throw away an otherwise complete code index.
