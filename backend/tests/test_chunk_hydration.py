"""Offline tests for hydrating ranked vector hits from persisted chunks."""

from pathlib import Path
from typing import Any

import pytest

from backend.db.connection import connect
from backend.db.repositories.chunk_sets import load_chunks_by_ids
from backend.embedding.models import VectorSearchHit
from backend.retrieval.chunk_hydration import (
    ChunkHydrationError,
    hydrate_vector_search_hits,
)


class RecordingChunkLoader:
    """Return configurable chunk records and capture batched loader calls."""

    def __init__(self, chunks: dict[str, dict[str, Any]]) -> None:
        """Configure the chunks visible to the hydration service.

        Args:
            chunks: Materialized chunk records keyed by stable chunk ID.

        Returns:
            None. Calls begin empty for later assertions.
        """
        self.chunks = chunks
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(
        self,
        chunk_set_id: str,
        chunk_ids: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        """Record one batched lookup and return configured matching chunks.

        Args:
            chunk_set_id: Expected chunk artifact supplied by hydration.
            chunk_ids: Ranked unique identifiers requested in one batch.

        Returns:
            Configured records whose IDs were requested.
        """
        self.calls.append((chunk_set_id, chunk_ids))

        # Match repository behavior by omitting unavailable identifiers.
        return {
            chunk_id: self.chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.chunks
        }


def _chunk(chunk_id: str, ordinal: int, text: str) -> dict[str, Any]:
    """Create one complete materialized chunk for hydration tests.

    Args:
        chunk_id: Stable identifier assigned to the test chunk.
        ordinal: Source-document order represented by the chunk.
        text: Exact chunk content returned by hydration.

    Returns:
        JSON-friendly chunk record matching the repository boundary.
    """
    return {
        "id": chunk_id,
        "source_document_id": "document-1",
        "ordinal": ordinal,
        "text": text,
        "character_start_offset": ordinal * 10,
        "character_end_offset": (ordinal + 1) * 10,
        "token_start_offset": ordinal * 2,
        "token_end_offset": (ordinal + 1) * 2,
        "page_start": ordinal + 1,
        "page_end": ordinal + 1,
        "section_path": ["Guide", f"Part {ordinal + 1}"],
        "source_metadata": {"block_ordinals": [ordinal]},
    }


@pytest.fixture
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect repository reads to one temporary initialized SQLite database.

    Args:
        tmp_path: Pytest-owned temporary directory.
        monkeypatch: Pytest helper that restores the production database path.

    Returns:
        Path to the isolated database used by the repository test.
    """
    database_path = tmp_path / "chunk-hydration.sqlite3"
    monkeypatch.setattr("backend.db.connection.DATABASE_PATH", database_path)

    # Opening one application connection initializes the production schema.
    with connect():
        pass
    return database_path


def _insert_repository_fixtures() -> None:
    """Persist chunks in two sets for scoped repository lookup tests.

    Args:
        None.

    Returns:
        None. Test rows are committed to the isolated application database.
    """
    timestamp = "2026-08-29T00:00:00Z"

    # Insert the required corpus, document, and parent artifact relationships.
    with connect() as connection:
        connection.execute(
            "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
            ("corpus-1", "Corpus", None, timestamp, timestamp),
        )
        connection.executemany(
            """
            INSERT INTO document (
                id, corpus_id, original_filename, storage_path, mime_type,
                size_bytes, content_sha256, uploaded_at
            ) VALUES (?, 'corpus-1', ?, ?, 'text/plain', 10, ?, ?)
            """,
            [
                ("document-1", "one.txt", "/test/one.txt", "sha-one", timestamp),
                ("document-2", "two.txt", "/test/two.txt", "sha-two", timestamp),
            ],
        )
        connection.executemany(
            """
            INSERT INTO chunk_set (
                id, corpus_id, fingerprint, chunking_config_json,
                chunker_name, chunker_version, status, chunk_count, created_at
            ) VALUES (?, 'corpus-1', ?, '{}', 'test', '1', 'ready', 1, ?)
            """,
            [
                ("chunk-set-1", "fingerprint-1", timestamp),
                ("chunk-set-2", "fingerprint-2", timestamp),
            ],
        )
        connection.executemany(
            """
            INSERT INTO chunk (
                id, chunk_set_id, source_document_id, ordinal, text,
                character_start_offset, character_end_offset,
                token_start_offset, token_end_offset, page_start, page_end,
                section_path_json, source_metadata_json
            ) VALUES (?, ?, ?, 0, ?, 0, 10, 0, 2, 1, 1, ?, ?)
            """,
            [
                (
                    "chunk-1",
                    "chunk-set-1",
                    "document-1",
                    "First text",
                    '["Guide","First"]',
                    '{"document":{"original_filename":"one.txt"}}',
                ),
                (
                    "foreign-chunk",
                    "chunk-set-2",
                    "document-2",
                    "Foreign text",
                    None,
                    '{"document":{"original_filename":"two.txt"}}',
                ),
            ],
        )


def test_hydrate_vector_search_hits_preserves_rank_and_provenance() -> None:
    """Verify hydration restores full chunks in vector-search order.

    Args:
        None.

    Returns:
        None. Assertions cover ranking, distances, text, and provenance.
    """
    chunks = {
        "chunk-1": _chunk("chunk-1", 0, "First text"),
        "chunk-2": _chunk("chunk-2", 1, "Second text"),
    }
    loader = RecordingChunkLoader(chunks)
    hits = (
        VectorSearchHit("chunk-2", 0.1),
        VectorSearchHit("chunk-1", 0.2),
    )

    hydrated = hydrate_vector_search_hits(hits, " chunk-set-1 ", loader)

    # The loader receives one normalized, ranked batch rather than per-hit queries.
    assert loader.calls == [("chunk-set-1", ("chunk-2", "chunk-1"))]
    assert [hit.rank for hit in hydrated] == [1, 2]
    assert [hit.chunk_id for hit in hydrated] == ["chunk-2", "chunk-1"]
    assert [hit.raw_distance for hit in hydrated] == [0.1, 0.2]
    assert hydrated[0].text == "Second text"
    assert hydrated[0].source_document_id == "document-1"
    assert hydrated[0].character_start_offset == 10
    assert hydrated[0].token_end_offset == 4
    assert hydrated[0].page_start == 2
    assert hydrated[0].section_path == ["Guide", "Part 2"]
    assert hydrated[0].source_metadata == {"block_ordinals": [1]}


def test_hydrate_vector_search_hits_skips_loading_for_empty_results() -> None:
    """Verify an empty search result returns immediately without SQLite work.

    Args:
        None.

    Returns:
        None. Assertions verify the loader remains unused.
    """
    loader = RecordingChunkLoader({})

    hydrated = hydrate_vector_search_hits((), "", loader)

    assert hydrated == ()
    assert loader.calls == []


def test_hydrate_vector_search_hits_rejects_duplicate_ids() -> None:
    """Verify duplicate vector IDs fail before ambiguous hydration begins.

    Args:
        None.

    Returns:
        None. Assertions verify the persistence loader remains unused.
    """
    loader = RecordingChunkLoader({"chunk-1": _chunk("chunk-1", 0, "Text")})
    hits = (
        VectorSearchHit("chunk-1", 0.1),
        VectorSearchHit("chunk-1", 0.2),
    )

    with pytest.raises(ChunkHydrationError, match="duplicate"):
        hydrate_vector_search_hits(hits, "chunk-set-1", loader)

    assert loader.calls == []


def test_hydrate_vector_search_hits_rejects_missing_or_foreign_chunks() -> None:
    """Verify incomplete scoped lookup results never produce partial context.

    Args:
        None.

    Returns:
        None. Assertions verify hydration fails for any unresolved hit.
    """
    loader = RecordingChunkLoader({"chunk-1": _chunk("chunk-1", 0, "Text")})
    hits = (
        VectorSearchHit("chunk-1", 0.1),
        VectorSearchHit("foreign-chunk", 0.2),
    )

    with pytest.raises(ChunkHydrationError, match="expected chunk set"):
        hydrate_vector_search_hits(hits, "chunk-set-1", loader)

    assert loader.calls == [
        ("chunk-set-1", ("chunk-1", "foreign-chunk")),
    ]


def test_load_chunks_by_ids_scopes_and_materializes_records(
    isolated_database: Path,
) -> None:
    """Verify one repository query omits foreign hits and decodes provenance.

    Args:
        isolated_database: Initialized temporary application database.

    Returns:
        None. Assertions cover chunk-set scoping and JSON materialization.
    """
    _insert_repository_fixtures()

    loaded = load_chunks_by_ids(
        "chunk-set-1",
        ("foreign-chunk", "chunk-1", "missing-chunk"),
    )

    # Only the requested chunk owned by the expected artifact may be returned.
    assert set(loaded) == {"chunk-1"}
    assert loaded["chunk-1"]["text"] == "First text"
    assert loaded["chunk-1"]["section_path"] == ["Guide", "First"]
    assert loaded["chunk-1"]["source_metadata"] == {
        "document": {"original_filename": "one.txt"}
    }
