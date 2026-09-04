# Answer Generation

The generation stage turns one persisted retrieval result into one immutable
answer. Groq is the first generation provider, but provider SDK objects remain
behind a backend-neutral interface so Gemini or another provider can be added
without changing pipeline orchestration or persistence.

## Configuration and Credentials

The backend exposes these Groq models:

| Model ID | Context window | Maximum completion | Notes |
| --- | ---: | ---: | --- |
| `openai/gpt-oss-20b` | 131,072 | 65,536 | Default production model. |
| `openai/gpt-oss-120b` | 131,072 | 65,536 | Larger production model. |
| `qwen/qwen3.6-27b` | 131,072 | 16,384 | Active Groq-hosted Qwen model. |
| `qwen/qwen3.8-27b` | 131,042 | 16,384 | Active Groq-hosted Qwen model. |

Create the ignored file `backend/.env` locally:

```env
GROQ_API_KEY=replace-with-your-key
```

The adapter loads this file without overriding an existing process environment
value. The key is never accepted in a run payload, persisted, returned, or
logged. A missing or rejected key fails only the run that reaches generation;
it does not prevent backend startup. Never commit this file or paste a live key
into source code, logs, documentation, or chat.

## Pipeline Flow

```text
retrieval completes
  -> atomically persist retrieval_result + retrieved_chunk rows
  -> advance pipeline_run to generation
  -> pack ranked chunks within the model context budget
  -> build the versioned source-labelled RAG prompt
  -> call Groq Chat Completions synchronously with stream=false
  -> validate answer, finish reason, usage, and provider provenance
  -> atomically persist generation_result + generation_context_chunk rows
  -> record generation duration and complete pipeline_run
```

The worker already runs synchronous pipeline work in a separate thread, so a
non-streaming Groq request does not block FastAPI's event loop. SDK retries are
disabled: one run makes one visible provider attempt.

## Prompt and Context Policy

`rag-answer-v1` keeps higher-priority instructions separate from the user
question. Retrieved chunks are labelled `[Source N]`, enclosed as untrusted
source data, and accompanied by document, page, and chunk identifiers. The
model is instructed to use only retrieved evidence, ignore instructions found
inside documents, cite its source labels, and admit insufficient context.

Prompt packing uses the same fixed backend tokenizer as chunking. It reserves
the requested output tokens and keeps ten percent of the advertised context
window unused to absorb differences between the backend tokenizer and the
model's private tokenizer. Complete chunks are added in retrieval-rank order.
Packing stops when the next chunk cannot fit; chunks are never silently
truncated and lower-ranked chunks never leapfrog an excluded higher rank.

The complete prompt is not duplicated in SQLite. It is reconstructable from
the run question, retrieval result, exact context links, prompt-template
version, provider-policy version, and immutable generation configuration. An
empty retrieval result skips Groq and persists a controlled
insufficient-context answer with zero token usage.

## Persistence and API

`generation_result` stores the answer, requested and provider-reported model,
effective generation settings, prompt/provider policy versions, finish reason,
optional token usage, request/fingerprint metadata, whether Groq was called,
and stage duration. `generation_context_chunk` stores the exact retrieval ranks
included in the prompt.

`GET /runs/{run_id}` exposes the hydrated ranked retrieval result and generation
state. After success it includes the answer, model provenance, usage, finish
reason, context links, and duration. A generation failure leaves the completed
retrieval result, chunk set, and vector index available while rolling back any
partial answer rows.

Generation logs contain provider/model identifiers, context count, finish
reason, provider-call decision, and duration. They never contain questions,
chunk text, prompts, answers, authorization headers, or API keys.

## Structured Failures

Generation distinguishes missing/rejected authentication, timeout, rate limit,
provider availability or capacity, oversized input, request rejection, invalid
response, tokenizer availability, and persistence failures. Raw Groq SDK
exceptions and response bodies are not returned to clients.

Groq's official references are the
[Chat Completions API](https://console.groq.com/docs/api-reference),
[model catalog](https://console.groq.com/docs/models), and
[error reference](https://console.groq.com/docs/errors).
