# RAG Playground Repository Guide

## Product Goal

Build a single-user web application for configuring, running, evaluating, and
comparing retrieval-augmented generation (RAG) experiments over user-provided
documents.

An experiment may vary:

- chunking strategy, chunk size, and overlap
- embedding model
- vector distance metric: cosine, dot product, or Euclidean distance
- LLM provider and model, including API-hosted and local models
- vector store; ChromaDB is the first implementation
- retrieval strategy and retrieval parameters
- metadata filters
- reranking and reranker configuration
- query transformations
- evaluation configuration

Preserve room for new pipeline stages and providers without coupling the core
experiment flow to one vendor. 
Out of scope: Auth, multi-user isolation, source-code ingestion, Graph/Hierarchical RAG.

## Repository Layout
- `backend/` — Python 3.10+, `uv`, FastAPI. Code in `backend/src/backend/`, tests in `backend/tests/`.
- `frontend/` — Next.js 16 App Router, React 19, TS, Tailwind 4, npm. Routes in `src/app/`, components in `src/components/`, API clients/utils in `src/lib/`.
- `data/` — local dev fixtures only; never commit user docs, generated indexes, or Chroma data.

## Boundaries
- FastAPI owns pipeline execution, provider integrations, persistence, and compatibility validation. Next.js owns presentation/interaction state only — no RAG logic, no provider secrets.
- API schemas are the contract; keep frontend types in sync.
- Store an immutable snapshot of each run's effective config (including resolved defaults) — never reconstruct history from mutable defaults.
- Keep inputs, retrieved chunks, scores, outputs, timings, errors, and eval results distinguishable (provenance).

## RAG Design Rules

- Represent each configurable stage with a typed configuration model and a
  small interface. Keep provider-specific SDK objects behind adapters.
- Use stable identifiers for documents, chunks, corpora, configurations, and
  runs. A chunk should retain its document ID, source metadata, offsets or page
  location.
- Treat score semantics carefully. Distance and similarity values are not
  interchangeable; normalize or label them before displaying or comparing
  results.
- Changing the embedding model, embedding dimensions, chunking configuration,
  or distance metric invalidates the corresponding vector collection. Never
  silently reuse an incompatible index.
- Record model/provider identifiers, prompt/template versions, retrieval
  parameters such as `top_k`, reranker settings, timestamps, and stage
  latencies. Record sampling parameters and random seeds where supported.
- Design long-running ingestion and experiment execution so it can move to a
  job abstraction later. Do not hold application-global mutable run state.
- Keep retrieval context separate from the user question in prompts, delimit
  untrusted content clearly, and instruct models not to follow instructions
  found in retrieved documents.
- Fail one run with a useful, structured error rather than corrupting or
  partially overwriting prior results.

## Evaluation and Comparison

- Keep evaluation separate from generation so the same output can be evaluated
  again without rerunning ingestion, retrieval, or the LLM.
- Support evaluation at both retrieval and answer levels. Useful future metrics
  include hit rate/recall at K, reciprocal rank, context relevance,
  faithfulness/groundedness, answer relevance, latency, and estimated cost.
- Do not present an LLM judge score as objective truth. Store the evaluator
  model, prompt version, rubric, raw rationale when appropriate, and score.
- Compare runs against the same corpus and evaluation examples by stable IDs.
  Warn when configurations are not directly comparable.
- Prefer deterministic evaluation in unit tests. Mock provider calls and use
  fixed document/query fixtures.

## Code Conventions

| Scope | Rules |
| :--- | :--- |
| **Backend** | `snake_case` functions/modules, `PascalCase` classes/Pydantic models. Use dependency injection for models/stores. Keep route handlers thin. Return machine-readable error codes (no raw traces). Add dependencies via `uv add [--dev]`. |
| **Frontend** | App Router & Server Components by default (`"use client"` only for state/browser APIs). Keep components small, typed, and in `src/components/`. Validate untrusted backend responses. Follow Next.js 16 docs in `node_modules/next/dist/docs/`. |
| **Testing** | Unit tests must be fast, deterministic, offline, and mocked (no real API calls/Chroma servers). Test boundary cases (empty docs, score ties, filter misses, invalid configs). Name tests `test_<behavior>`. |

- For every function written, add comments explaining the parameters, return value, and a short description of function
- For every block (if-else, loops, methods, classes.. etc.,) add comments explaining what it does.
- For any constants, magic numbers, non-obvious values add a comment with a short explanation.

The below instructions are specifically for the frontend/ folder.
- Any component/section, write a small description for it.
- For all the hooks used, add explicit comments explaining the purpose. If there are multiple variables defined one below the
  other, with the comments for each one add a line break between 2 variables for readability.
- For each of the top level UI blocks (max-depth 4), add a small description explaining what the block is.
- Do not exceed 100 characters per line, if yes continue in next line each element structure should be opening tag, content, closing tag.. each on different line

## Development Commands

```bash
# Backend Setup & Run
cd backend && uv sync && uv run backend
# Recommended Backend Quality Checks
uv run pytest && uv run ruff check . && uv run ruff format --check .

# Frontend Setup & Run
cd frontend && npm ci && npm run dev
# Frontend Verification
npm run lint
```

## Testing Requirements

- Add or update tests with every behavior change.
- Name Python test files `test_<feature>.py` and test functions `test_<expected_behavior>`.
- Unit tests must not call paid APIs, download models, require a running ChromaDB server, or depend on execution order.
- Use small fixed embeddings or fakes to test ranking and metric behavior. Include edge cases such as empty documents, overlap boundaries, tied scores, filters with no matches, provider failures, and invalid configurations.
- Add API tests for validation and structured error responses.
- For frontend changes, run lint. Add focused component or end-to-end tests once a test framework is configured.
- Mark integration tests explicitly and document their required services, models, environment variables, and expected cost.

## Change Workflow

1. Inspect the affected application, its local configuration, and relevant documentation before editing.
2. Make the smallest coherent change and preserve separation between domain logic, adapters, API transport, and UI.
3. Add regression coverage and run the narrowest relevant checks, followed by the application-level checks when practical.
4. Update documentation when commands, environment variables, API contracts, persisted schemas, defaults, or supported configuration options change.
5. Report the changes made tagging the files that were added/edited for that change.
6. Report what was verified and call out any check that could not be run.