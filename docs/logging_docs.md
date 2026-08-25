# Backend Logging

The backend uses Python's standard `logging` package. Application records are
written to stdout so local Uvicorn output, containers, and future log collectors
can consume the same stream.

## Configuration

`backend.logging_config.configure_logging()` initializes the `backend` logger
hierarchy when the FastAPI application module loads. The default level is
`INFO`.

Set `BACKEND_LOG_LEVEL` before starting the server to change application
verbosity:

```bash
BACKEND_LOG_LEVEL=DEBUG uv run backend
```

Supported values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
Values are case-insensitive. An unsupported value falls back to `INFO`.

Each record contains:

```text
timestamp | level | module | event and key=value fields
```

Uvicorn retains its own access and server lifecycle loggers. The application
configuration is limited to the `backend` hierarchy to avoid duplicate Uvicorn
records.

## Stage Events

Upload and parsing emit:

- `upload_started`
- `upload_file_validated`
- `parsing_started`
- `parsing_completed`
- `upload_completed`
- `upload_rejected`, `parsing_failed`, `upload_failed`, or `upload_rolled_back`

Chunking emits:

- `chunk_set_requested`
- `chunk_set_build_started`
- `document_chunking_started`
- `document_chunking_completed`
- `chunk_set_completed`
- `chunk_set_reused` or `chunk_set_reused_after_race`
- `chunk_set_rejected`, `chunk_set_persistence_failed`, or
  `chunk_set_readback_failed`

Pipeline execution emits:

- `pipeline_run_requested`
- `pipeline_run_created`
- `pipeline_run_started`
- `pipeline_run_completed`
- `pipeline_run_rejected`, `pipeline_run_stage_failed`, or
  `pipeline_run_failed`

Normal lifecycle transitions use `INFO`. Invalid input and expected rejected
operations use `WARNING`. Unexpected operational failures use `ERROR` with an
exception traceback.

## Logged Data

Stage logs contain bounded operational fields such as:

- corpus, document, run, and chunk-set IDs;
- safely represented filenames;
- parser and chunking strategy names;
- file, character, page, block, warning, and chunk counts;
- stage durations and artifact reuse decisions; and
- stable error codes.

Logs must not include document text, chunk text, user questions, complete API
payloads, tokenizer objects, API keys, authorization headers, or provider
secrets. Filenames use their Python representation so control characters cannot
silently create additional log lines.
