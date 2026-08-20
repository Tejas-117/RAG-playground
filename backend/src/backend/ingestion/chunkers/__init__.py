"""Public contracts for reusable canonical-document chunking."""

from backend.ingestion.chunkers.models import ChunkSpan
from backend.ingestion.chunkers.strategies import (
    MAX_CHUNK_CHARACTERS,
    get_chunker,
)
from backend.ingestion.chunkers.tokenizer import get_chunking_tokenizer

__all__ = [
    "MAX_CHUNK_CHARACTERS",
    "ChunkSpan",
    "get_chunker",
    "get_chunking_tokenizer",
]
