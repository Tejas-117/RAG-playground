import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import (
    corpora,
    indexes,
    pipeline_options,
    runs,
    testing,
    uploads,
)
from backend.db.repositories.prepared_indexes import (
    fail_interrupted_prepared_indexes,
)
from backend.db.repositories.runs import fail_interrupted_runs
from backend.logging_config import configure_logging
from backend.pipeline.worker import PipelineRunWorker

# Configure application loggers before routes begin handling requests.
configure_logging()


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Recover interrupted jobs and own the local background worker lifecycle.

    Args:
        application: FastAPI application entering or leaving its lifespan.

    Yields:
        Control while the API and background worker are available.
    """
    # Terminalize work abandoned by a previous process before claiming queued jobs.
    fail_interrupted_prepared_indexes()
    fail_interrupted_runs()
    worker = PipelineRunWorker()
    worker_task = asyncio.create_task(worker.run_forever())

    try:
        # Serve requests while the independently persisted worker processes runs.
        yield
    finally:
        # Cancel the polling coroutine; active state remains recoverable in SQLite.
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task


app = FastAPI(title="RAG Playground API", lifespan=application_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(uploads.router)
app.include_router(corpora.router)
app.include_router(pipeline_options.router)
app.include_router(indexes.router)
app.include_router(runs.router)
app.include_router(testing.router)


def main() -> None:
    """Start the FastAPI development server.

    Returns:
        None. The function blocks while the development server is running.
    """
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
