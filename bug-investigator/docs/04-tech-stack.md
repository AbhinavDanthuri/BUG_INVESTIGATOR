# 04 · Tech Stack

| Layer | Choice | Why | Alternative |
|-------|--------|-----|-------------|
| Language | Python 3.11+ | LangGraph, FAISS, sentence-transformers are all Python-first | — |
| API | FastAPI + Uvicorn | Async, typed, auto docs at `/docs`; matches the resume line | Flask |
| Agent orchestration | LangGraph | Explicit graph with conditional edges is exactly the critic loop | Plain Python loop |
| LLM access | LangChain chat model wrappers behind a small `LLM` interface | Swap providers without touching agent code | Direct SDKs |
| LLM (default) | Google Gemini Flash via a free AI Studio key (`LLM_PROVIDER=gemini`, `LLM_MODEL` env var) | No cost; rate limits handled with retry/backoff and small prompts | Anthropic, OpenAI, Ollama (free, local) |
| Embeddings | sentence-transformers, `BAAI/bge-small-en-v1.5` (configurable) | Runs locally, no per-call cost, small enough for a laptop | `jina-embeddings-v2-base-code` (code-specific, heavier) |
| Vector search | FAISS (`IndexFlatIP` + `IndexIDMap`) | Exact search is fine at repo scale (tens of thousands of chunks) | HNSW later if needed |
| Sparse search | `rank-bm25` | Catches identifiers and error strings dense search misses | Elasticsearch (overkill) |
| Database | MySQL 8 | On the resume; JSON columns cover flexible agent payloads | Postgres |
| ORM / migrations | SQLAlchemy 2.0 + Alembic (added at M3) | Typed models; `schema.sql` is generated from them | Raw SQL |
| GitHub | `httpx` thin client + `git clone` | Control over ETags/rate limits, full history via git | PyGithub |
| Frontend | React + Vite | Fast dev loop, minimal setup | Next.js |
| Testing | pytest, fake LLM + fake retriever | Graph logic tested with no API calls | — |
| Lint/format | ruff | One tool | black + flake8 |
| Infra | Docker Compose | One command for MySQL + API | — |

## Config (env vars)
`DATABASE_URL`, `GITHUB_TOKEN`, `LLM_PROVIDER` (`gemini` | `anthropic` | `openai` | `ollama`), `LLM_MODEL`, `GOOGLE_API_KEY`, `EMBEDDING_MODEL`, `DATA_DIR`, `MAX_ITERATIONS`, `TOKEN_BUDGET`. See `.env.example`.
