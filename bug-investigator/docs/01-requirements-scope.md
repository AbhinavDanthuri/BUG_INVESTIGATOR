# 01 · Requirements & Scope

## Problem
Triaging a bug report means hunting through code, docs, old issues and commit history to guess where the bug lives. The Autonomous Bug Investigator takes a GitHub issue URL, gathers that evidence itself, proposes a root cause, and suggests a fix plus tests. Every claim has to point back to evidence.

## Users
- Maintainers and contributors triaging incoming issues.
- Recruiters/interviewers looking at a live demo (this is also a portfolio project, so it has to be demoable and measurable).

## Functional requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Register a public GitHub repo and index it: code, docs (README/md), issues + PRs, commit history. Re-index only when the HEAD commit changes. |
| FR-2 | Accept a GitHub issue URL and start an investigation asynchronously. |
| FR-3 | Run a LangGraph workflow: **planner → investigator → fix → critic**. If the critic rejects, loop back for another pass (max N iterations, default 3). |
| FR-4 | Every root-cause claim cites evidence (`path:lines`, issue `#n`, or commit SHA). Claims with no valid citation are rejected by the critic. |
| FR-5 | Produce a report: root cause, confidence, affected files, evidence list, suggested diff, suggested test cases, exportable as Markdown. |
| FR-6 | Persist the full run in MySQL: every agent step, every retrieval, every critic review, token usage. A run can be replayed from history. |
| FR-7 | React UI: submit an issue, watch progress, read the report, browse past runs. |
| FR-8 | Eval harness: replay closed issues that have a merged fix PR and score file-level localization (hit@1, hit@3, recall) against a BM25-only baseline. |

## Non-functional requirements
- **Latency:** typical investigation under ~3 minutes on a mid-size repo (target, to be measured).
- **Cost control:** token budget per run (config), hard stop when exceeded.
- **Safety:** issue text and repo content are untrusted input (prompt injection). The tool is read-only: it never executes repo code and never pushes anything to GitHub.
- **Reproducibility:** store model name, embedding model, prompt version and indexed commit SHA with each run.
- **Rate limits:** GitHub token required, conditional requests/caching for issues.
- **Local-first:** `docker compose up` gives MySQL + API; frontend via `npm run dev`.

## Scope

**In scope (MVP)**
- Public GitHub repos, any language (text-based chunking; Python/Java used for evaluation).
- Hybrid retrieval (FAISS dense + BM25, fused with RRF) over code, docs, issues, commits.
- 4-agent LangGraph loop with critic feedback.
- MySQL persistence, FastAPI backend, React UI.
- Eval harness on a small set of closed issues.

**Out of scope (MVP)**
- Private repos / GitHub App auth, multi-user accounts.
- Running the suggested fix or tests in a sandbox.
- Opening PRs or commenting on issues automatically.
- Symbol-level (tree-sitter) chunking — planned as an upgrade after the baseline works.

**Stretch**
- Sandbox execution of generated tests to validate the fix.
- Live progress over SSE instead of polling.
- Symbol-aware chunking and call-graph hints.

## Success criteria
1. End-to-end run works on at least 2 repos the tool has never seen.
2. Report always contains cited evidence; ungrounded answers get bounced by the critic.
3. Eval: file-level hit@3 beats the BM25-only baseline on the same issues. The actual numbers get measured, then written on the resume.
4. A recruiter can open the demo, paste an issue URL, and see a report with a visible critic loop.

## Milestones
| # | Milestone | State |
|---|-----------|-------|
| M0 | Requirements, architecture, data model, scaffold | done |
| M1 | Ingestion (clone, issues, commits) + chunking + FAISS/BM25 retrieval | done |
| M2 | Agent loop, prompts, critic gating and citation guardrails | done |
| M3 | Runner + MySQL persistence + API + React UI | done |
| M4 | Eval harness and BM25 baseline | built; needs a real run to produce numbers |

Remaining work is prompt tuning against real repositories and running the eval to get figures.

## Assumptions I made (change any of these)
- LLM provider is pluggable; default Gemini Flash on a free AI Studio key, with Anthropic/OpenAI/Ollama also wired in.
- Embeddings run locally with sentence-transformers (no API cost); `EMBEDDING_MODEL=hashing` gives a download-free fallback for tests.
- Evaluate on 2–3 small/medium open-source Python repos with well-labelled closed issues.

## Risks
- LLM JSON output is flaky → strict parsing + one repair retry.
- Large repos blow up indexing time → cap file size/count, skip vendored/binary files.
- Eval ground truth (files changed in the fix PR) is noisy → use only issues linked to a single small PR.
