# Answer Generation

The generation stage turns one persisted retrieval result into one immutable
answer. Groq and Ollama remain behind the same backend-neutral provider interface.

## Configuration and Credentials

The backend exposes these generation models:

| Provider | Model ID | Context budget | Maximum completion | Notes |
| --- | --- | ---: | ---: | --- |
| Groq | `openai/gpt-oss-20b` | 131,072 | 65,536 | Default hosted model. |
| Groq | `openai/gpt-oss-120b` | 131,072 | 65,536 | Larger hosted model. |
| Groq | `qwen/qwen3.6-27b` | 131,072 | 16,384 | Hosted Qwen model. |
| Groq | `qwen/qwen3.8-27b` | 131,042 | 16,384 | Hosted Qwen model. |
| Ollama | `llama3.2:1b` | 8,192 | 2,048 | Conservative local budget. |
| Ollama | `llama3.2:3b` | 8,192 | 2,048 | Conservative local budget. |

Create the ignored file `backend/.env` locally:

```env
GROQ_API_KEY=replace-with-your-key
```

The adapter loads this file without overriding an existing process environment
value. The key is never accepted in a run payload, persisted, returned, or
logged. A missing or rejected key fails only the run that reaches generation;
it does not prevent backend startup. Never commit this file or paste a live key
into source code, logs, documentation, or chat.

Ollama generation calls `POST /api/chat` with `stream: false`. The backend does
not install models or start Ollama. Install the selected models separately:

```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

Both embedding and generation use `OLLAMA_BASE_URL`, defaulting to
`http://localhost:11434`. A missing model or unavailable service fails the active
benchmark with a structured error.

## Pipeline Flow

```text
benchmark retrieval completes
  -> atomically persist the child retrieval result and ranked chunks
  -> advance the benchmark child to generation
  -> pack ranked chunks within the model context budget
  -> build the versioned source-labelled RAG prompt
  -> call the selected Groq or Ollama adapter synchronously with stream=false
  -> validate answer, finish reason, usage, and provider provenance
  -> atomically persist the child generation result and context links
  -> record generation duration and complete the benchmark child
```

The worker already runs synchronous pipeline work in a separate thread, so a
non-streaming provider request does not block FastAPI's event loop. Automatic
retries are disabled: one example makes one visible provider attempt.

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
empty retrieval result skips the selected provider and persists a controlled
insufficient-context answer with zero token usage.

## Persistence and API

`benchmark_generation_result` stores each answer and its requested and
provider-reported model,
effective generation settings, prompt/provider policy versions, finish reason,
optional token usage, request/fingerprint metadata, whether a provider was called,
and stage duration. `benchmark_generation_context_chunk` stores the exact
retrieval ranks included in the prompt.

`GET /runs/{run_id}` exposes the hydrated ranked retrieval result and generation
state. After success it includes the answer, model provenance, usage, finish
reason, context links, and duration. A generation failure leaves the completed
retrieval result and prepared-index provenance available while rolling back any
partial answer rows.

Generation logs contain provider/model identifiers, context count, finish
reason, provider-call decision, and duration. They never contain questions,
chunk text, prompts, answers, authorization headers, or API keys.

## Structured Failures

Generation distinguishes missing/rejected authentication, timeout, rate limit,
provider availability or capacity, oversized input, request rejection, invalid
response, tokenizer availability, and persistence failures. Raw provider
exceptions and response bodies are not returned to clients.

Groq's official references are the
[Chat Completions API](https://console.groq.com/docs/api-reference),
[model catalog](https://console.groq.com/docs/models), and
[error reference](https://console.groq.com/docs/errors).
Ollama documents its request and response shape in the
[chat API reference](https://docs.ollama.com/api/chat).
