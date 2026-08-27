# RAG Pipeline Parameters

This document defines the user-configurable parameters for the RAG Playground MVP and identifies controls reserved for later versions.

The MVP exposes parameters that create meaningful, understandable experiment differences. Other values use sensible backend defaults and are captured in each immutable run configuration snapshot.

## 1. Corpus Upload

### MVP Parameters

- **Corpus name**: required.
- **Files**: required; limited to supported formats.
- **Description**: optional short description.

### Reserved for Later Versions

- Tags and custom metadata.
- Per-document metadata editing.
- Document-level inclusion/exclusion.
- OCR language and options.
- Corpus-level access/sharing controls.

Corpora are immutable. A user creates a new corpus to change the document set or document content.

## 2. Parsing

### MVP Parameters

No user-configurable parsing parameters initially. The backend selects a parser based on file type and uses documented defaults.

The UI should display the detected file type, parser used, parse status, page/character count, and parse warnings.

### Reserved for Later Versions

- Parser selection.
- OCR enablement and language.
- PDF extraction mode.
- Table extraction.
- Header/footer removal.
- Custom cleanup rules.

Parsing controls are deferred until the application supports enough parsers and file types for the choices to be useful.

## 3. Chunking

Chunking is the main configurable ingestion stage in the MVP.

### MVP Parameters

- **Chunking strategy**:
  - Recursive (default).
  - Fixed-size with overlap.
  - Paragraph.
- **Chunk size**: expressed in tokens.
- **Chunk overlap**: applicable to recursive and fixed-size strategies.
- **Chunk preview**: an optional preview of representative chunks before running; it is not a persisted pipeline parameter.

Recommended defaults:

```text
Strategy: recursive
Chunk size: 800 tokens
Overlap: 100 tokens
```

Paragraph chunking always resolves overlap to `0`. All strategies enforce an
internal 32,000-character safety limit in addition to the selected token limit.
The character cap is a backend guardrail rather than a user-configurable value.

### Reserved for Later Versions

- Semantic chunking.
- Late chunking.
- Sentence/proposition chunking.
- Parent-child chunking.
- Markdown/HTML/code-aware chunking controls.
- Custom separator hierarchy.
- Minimum/maximum chunk size.
- Tokenizer selection.
- Per-document chunking overrides.

The MVP language should remain simple: users should understand the effect of strategy, size, and overlap without requiring tokenizer knowledge.

## 4. Embedding and Vector Index

### MVP Parameters

- **Embedding provider**: only providers configured and available in the backend.
- **Embedding model**.
- **Distance metric**:
  - Cosine (default).
  - Dot product.
  - Euclidean.

The backend validates compatibility before building an index. It must not silently reuse an incompatible index.

The first adapter calls Ollama's `/api/embed` HTTP endpoint. The backend does
not install, pull, inspect, or start models. `OLLAMA_BASE_URL` may point to a
local or remote service. Provider and model failures are recorded on the run,
and future providers can implement the same backend-neutral adapter contract.
Chroma stores the explicit vectors locally; it never chooses or invokes the
embedding model.

### Reserved for Later Versions

- Vector-store selection beyond Chroma.
- Embedding batch size.
- Device selection, such as CPU/GPU.
- Vector-index algorithm/tuning options.
- Quantization.
- Multi-vector embeddings.
- Hybrid keyword/vector indexing.
- Custom embedding endpoint settings in the experiment UI.

For the MVP, Chroma is an implementation detail: users choose embedding behavior, not the database engine.

## 5. Query Transformation

### MVP Parameters

No user-configurable transformation: use the raw user query.

### Reserved for Later Versions

- Query rewriting.
- Multi-query retrieval.
- HyDE.
- Query decomposition.
- Conversational query contextualization.
- Query-expansion model and prompt selection.

Deferring transformations keeps early retrieval comparisons clear and attributable to the original query.

## 6. Retrieval

### MVP Parameters

- **Top K**: number of chunks to retrieve.
- **Minimum relevance threshold**: optional; expose only if score normalization is reliable for the selected metric/model.

Recommended default:

```text
Top K: 10
```

The UI should show each retrieved chunk with rank, source document/page, clearly
labelled raw or normalized score, and chunk text.

### Reserved for Later Versions

- Metadata filters.
- Hybrid retrieval and keyword/vector weighting.
- Maximum marginal relevance (MMR).
- Diversity controls.
- Per-document retrieval limits.
- Retrieval timeout/budget.
- Multi-index retrieval.
- Dynamic `top_k`.

## 7. Reranking

### MVP Parameters

Reranking is disabled for the first MVP and has no user-facing configuration.

### Reserved for Later Versions

- Enable/disable reranking.
- Reranker provider/model.
- Number of candidates passed to the reranker.
- Number retained after reranking.
- Score threshold.
- Cross-encoder versus API reranker selection.

Reranking introduces another model, latency, cost, and score semantics, so it should follow basic retrieval visibility and validation.

## 8. Generation

### MVP Parameters

- **LLM provider**.
- **LLM model**.
- **Temperature**.
- **Maximum output tokens**: optional.

Recommended defaults:

```text
Temperature: 0.2
Maximum output tokens: backend-defined safe default
```

The prompt template is backend-controlled and versioned. The UI may show its name/version but should not offer a free-form prompt editor initially.

### Reserved for Later Versions

- Prompt-template selection.
- Custom prompt editing.
- Top-p, frequency penalty, and presence penalty.
- Seed, where supported.
- Streaming mode controls.
- Structured-output schemas.
- Tool calling.
- Answer language/style controls.
- Context compression/summarization.
- Multiple generation candidates.

Holding the prompt steady in the MVP makes chunking, embedding, and retrieval comparisons more meaningful.

## 9. Evaluation

### MVP Parameters

Keep evaluation optional and simple:

- **Retrieval metrics**: select any combination of hit rate at K, recall at K,
  and reciprocal rank when labelled relevant documents exist.
- **Answer metrics**: select any combination of groundedness, answer relevance,
  and answer correctness when their required inputs exist.

The query remains required, but both metric lists may be empty. Empty lists mean
the run produces retrieval and generation output without evaluating it. New
single-question experiments initially select groundedness and answer relevance;
users may clear both selections. Retrieval metrics and answer correctness require
dataset annotations and are unavailable for an ad hoc question.

Evaluation configuration is stored with the immutable run snapshot. Metric
execution and evaluation-result artifacts remain unavailable until the
evaluation pipeline is implemented.

### Reserved for Later Versions

- Custom evaluation datasets and annotations.
- LLM-judge model selection.
- Evaluator prompt/rubric selection.
- Per-metric thresholds.
- Human feedback workflows.
- Cost estimation.
- A/B comparison dashboards.
- Batch evaluation schedules.
- Regression gates.

Evaluation remains separate from generation so the same output can be evaluated with another rubric or evaluator without rerunning ingestion, retrieval, or generation.

## MVP Configuration Surface

```text
Corpus
  Corpus: selected immutable corpus

Chunking
  Strategy: recursive | fixed-size | paragraph
  Chunk size: 800
  Overlap: 100

Embeddings
  Provider: ...
  Model: ...
  Metric: cosine | dot product | euclidean

Retrieval
  Top K: 10

Generation
  Provider: ...
  Model: ...
  Temperature: 0.2
  Max output tokens: ...

Evaluation
  Enable evaluation: on/off
  Evaluation set: ... # only when implemented
```

The governing rule is to expose a control only when a user can understand its trade-off and compare its impact. Controls requiring provider-specific knowledge, producing unstable comparisons, or lacking meaningful implementation choices remain backend defaults until a later version.
