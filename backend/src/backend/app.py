import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import corpora, uploads

app = FastAPI(title="RAG Playground API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(uploads.router)
app.include_router(corpora.router)


def main() -> None:
    """Start the FastAPI development server.

    Returns:
        None. The function blocks while the development server is running.
    """
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
