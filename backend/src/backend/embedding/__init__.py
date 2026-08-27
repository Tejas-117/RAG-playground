"""Provider-neutral document embedding and vector-index contracts."""

from backend.embedding.models import (
    EmbeddingBatch,
    EmbeddingInputPurpose,
    EmbeddingProvider,
    VectorStore,
)
from backend.embedding.service import (
    VectorIndexBuildResult,
    build_or_reuse_vector_index,
)

__all__ = [
    "EmbeddingBatch",
    "EmbeddingInputPurpose",
    "EmbeddingProvider",
    "VectorIndexBuildResult",
    "VectorStore",
    "build_or_reuse_vector_index",
]
