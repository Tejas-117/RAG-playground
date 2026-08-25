"""Tests for fingerprinted chunk-set persistence, reuse, and provenance."""

import re
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.db.connection import connect
from backend.db.repositories import chunk_sets as chunk_set_repository
from backend.ingestion.chunk_sets import (
    ChunkingCorpusNotFoundError,
    EmptyChunkingCorpusError,
    MissingParseArtifactError,
    build_or_reuse_chunk_set,
)
from backend.ingestion.chunkers.models import TokenizedText, TokenOffset
from backend.pipeline.configs import ChunkingConfig, ChunkingStrategy


class CountingTokenizer:
    """Provide deterministic test offsets and expose encoding work for reuse checks."""

    identifier = "test-tokenizer"
    revision = "1"
    special_tokens_policy = "none"

    def __init__(self, asset_sha256: str = "digest-one") -> None:
        """Create a fake tokenizer with a selectable fingerprint identity.

        Args:
            asset_sha256: Test digest included in chunk-set compatibility.

        Returns:
            None. The instance begins with no encoding calls.
        """
        self.asset_sha256 = asset_sha256
        self.encode_count = 0

    def encode(self, text: str) -> TokenizedText:
        """Measure non-whitespace test tokens and count the operation.

        Args:
            text: Canonical source text to tokenize.

        Returns:
            Ordered source offsets for every non-whitespace sequence.
        """
        self.encode_count += 1

        # Return source-aware offsets without using a model or network service.
        return TokenizedText(
            offsets=tuple(
                TokenOffset(match.start(), match.end())
                for match in re.finditer(r"\S+", text)
            )
        )


@pytest.fixture
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all repository connections to one temporary SQLite database.

    Args:
        tmp_path: Pytest-owned temporary directory.
        monkeypatch: Pytest helper for restoring the configured database path.

    Returns:
        Path to the isolated initialized database.
    """
    database_path = tmp_path / "chunk-sets.sqlite3"
    monkeypatch.setattr("backend.db.connection.DATABASE_PATH", database_path)

    # Opening one application connection initializes the production schema.
    with connect():
        pass
    return database_path


def _insert_corpus(corpus_id: str = "corpus-1") -> None:
    """Insert one known corpus for chunk-set service tests.

    Args:
        corpus_id: Stable identifier assigned to the test corpus.

    Returns:
        None. The row is committed to the active isolated database.
    """
    # Use fixed timestamps so ordering and fingerprints remain deterministic.
    with connect() as connection:
        connection.execute(
            "INSERT INTO corpus VALUES (?, ?, ?, ?, ?)",
            (
                corpus_id,
                "Test corpus",
                None,
                "2026-08-14T00:00:00Z",
                "2026-08-14T00:00:00Z",
            ),
        )


def _insert_document_with_parse(
    document_id: str,
    parse_id: str,
    text: str,
    uploaded_at: str,
    pages: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int]],
) -> None:
    """Insert one source document, canonical parse, pages, and blocks.

    Args:
        document_id: Stable source-document identifier.
        parse_id: Stable canonical artifact identifier.
        text: Persisted canonical text.
        uploaded_at: Timestamp controlling deterministic corpus order.
        pages: Page number and half-open character ranges.
        blocks: Block ordinal and half-open character ranges.

    Returns:
        None. All source rows are committed together.
    """
    with connect() as connection:
        connection.execute(
            "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_id,
                "corpus-1",
                f"{document_id}.txt",
                f"uploads/{document_id}.txt",
                "text/plain",
                len(text.encode("utf-8")),
                document_id.ljust(64, "0"),
                uploaded_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO document_parse VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                parse_id,
                document_id,
                text,
                len(text.encode("utf-8")),
                len(text),
                "test-parser",
                "1.0.0",
                '{"language":"en"}',
                "[]",
                len(pages),
                len(blocks),
                1,
                "2026-08-14T00:00:01Z",
            ),
        )
        page_ids: list[str] = []

        # Persist physical page offsets before blocks reference those rows.
        for page_number, character_start, character_end in pages:
            page_id = f"{parse_id}-page-{page_number}"
            page_ids.append(page_id)
            connection.execute(
                "INSERT INTO parsed_page VALUES (?, ?, ?, ?, ?, ?)",
                (
                    page_id,
                    parse_id,
                    page_number,
                    character_start,
                    character_end,
                    "{}",
                ),
            )

        # Attach each block to the first page whose half-open range intersects it.
        for ordinal, character_start, character_end in blocks:
            page_id = next(
                (
                    page_ids[index]
                    for index, (_, page_start, page_end) in enumerate(pages)
                    if character_start < page_end and page_start < character_end
                ),
                None,
            )
            connection.execute(
                "INSERT INTO parsed_block VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{parse_id}-block-{ordinal}",
                    parse_id,
                    page_id,
                    ordinal,
                    ordinal,
                    character_start,
                    character_end,
                    None,
                    "{}",
                ),
            )


def _seed_complete_corpus() -> None:
    """Create two parsed documents with page and block provenance.

    Args:
        None.

    Returns:
        None. The isolated database receives deterministic source fixtures.
    """
    _insert_corpus()
    _insert_document_with_parse(
        "document-a",
        "parse-a",
        "alpha beta gamma",
        "2026-08-14T00:00:01Z",
        [(1, 0, 10), (2, 11, 16)],
        [(0, 0, 5), (1, 11, 16)],
    )
    _insert_document_with_parse(
        "document-b",
        "parse-b",
        "delta epsilon",
        "2026-08-14T00:00:02Z",
        [(1, 0, 13)],
        [(0, 0, 13)],
    )


def test_build_chunk_set_persists_provenance_and_reuses_fingerprint(
    isolated_database: Path,
) -> None:
    """Verify deterministic chunks, per-document ordinals, and ready reuse.

    Args:
        isolated_database: Initialized temporary application database.

    Returns:
        None. Assertions cover artifact identity, provenance, and reuse.
    """
    _seed_complete_corpus()
    tokenizer = CountingTokenizer()
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size_tokens=2,
        chunk_overlap_tokens=0,
    )
    with patch("backend.ingestion.chunk_sets.logger") as chunk_logger:
        first = build_or_reuse_chunk_set("corpus-1", config, tokenizer)
        second = build_or_reuse_chunk_set("corpus-1", config, tokenizer)

    # Identical inputs return the one persisted artifact without re-encoding documents.
    assert first.reused is False
    assert second.reused is True
    assert first.artifact["id"] == second.artifact["id"]
    assert first.artifact["fingerprint"] == second.artifact["fingerprint"]
    assert tokenizer.encode_count == 2
    assert first.artifact["chunk_count"] == 3
    assert [
        (chunk["source_document_id"], chunk["ordinal"])
        for chunk in first.artifact["chunks"]
    ] == [
        ("document-a", 0),
        ("document-a", 1),
        ("document-b", 0),
    ]

    # Page, parse, and block provenance comes from half-open range intersections.
    first_chunk = first.artifact["chunks"][0]
    assert first_chunk["text"] == "alpha beta"
    assert (first_chunk["page_start"], first_chunk["page_end"]) == (1, 1)
    assert first_chunk["source_metadata"]["parse_id"] == "parse-a"
    assert first_chunk["source_metadata"]["block_ordinals"] == [0]
    assert first_chunk["section_path"] is None

    # Reuse leaves exactly one ready parent row and preserves deterministic chunk IDs.
    with sqlite3.connect(isolated_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunk_set").fetchone()[0] == 1
    assert [chunk["id"] for chunk in first.artifact["chunks"]] == [
        chunk["id"] for chunk in second.artifact["chunks"]
    ]

    # The service records both the initial build and the fingerprint reuse decision.
    event_templates = [call.args[0] for call in chunk_logger.info.call_args_list]
    assert any("chunk_set_completed" in event for event in event_templates)
    assert any("chunk_set_reused" in event for event in event_templates)


def test_chunk_set_fingerprint_changes_with_config_and_tokenizer(
    isolated_database: Path,
) -> None:
    """Verify every boundary-affecting input creates a distinct artifact.

    Args:
        isolated_database: Initialized temporary application database.

    Returns:
        None. Assertions compare configuration and tokenizer identities.
    """
    _seed_complete_corpus()
    baseline = build_or_reuse_chunk_set(
        "corpus-1",
        ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size_tokens=2,
            chunk_overlap_tokens=0,
        ),
        CountingTokenizer(),
    )
    changed_config = build_or_reuse_chunk_set(
        "corpus-1",
        ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size_tokens=1,
            chunk_overlap_tokens=0,
        ),
        CountingTokenizer(),
    )
    changed_tokenizer = build_or_reuse_chunk_set(
        "corpus-1",
        ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size_tokens=2,
            chunk_overlap_tokens=0,
        ),
        CountingTokenizer(asset_sha256="digest-two"),
    )

    # Compatibility changes must never silently point to the baseline artifact.
    assert (
        len(
            {
                baseline.artifact["fingerprint"],
                changed_config.artifact["fingerprint"],
                changed_tokenizer.artifact["fingerprint"],
            }
        )
        == 3
    )


def test_concurrent_chunk_set_save_reuses_ready_winner(
    isolated_database: Path,
) -> None:
    """Verify a uniqueness race converges on the concurrently saved artifact.

    Args:
        isolated_database: Initialized temporary application database.

    Returns:
        None. Assertions verify the race result and persisted artifact count.
    """
    _seed_complete_corpus()

    def persist_then_raise(
        chunk_set: dict[str, object],
        chunks: list[dict[str, object]],
    ) -> None:
        """Simulate another request committing immediately before this save.

        Args:
            chunk_set: Complete parent artifact prepared by the service.
            chunks: Ordered child chunks prepared by the service.

        Returns:
            None. The simulated losing request receives an integrity error.
        """
        # Persist through the real repository as if the concurrent winner committed.
        chunk_set_repository.save_ready_chunk_set(chunk_set, chunks)
        raise sqlite3.IntegrityError("simulated fingerprint race")

    # Replace only the imported save boundary used by the orchestration service.
    with patch(
        "backend.ingestion.chunk_sets.save_ready_chunk_set",
        side_effect=persist_then_raise,
    ):
        result = build_or_reuse_chunk_set(
            "corpus-1",
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size_tokens=2,
                chunk_overlap_tokens=0,
            ),
            CountingTokenizer(),
        )

    # The losing request returns the winner instead of exposing a uniqueness error.
    assert result.reused is True
    assert result.artifact["status"] == "ready"
    with sqlite3.connect(isolated_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunk_set").fetchone()[0] == 1


def test_chunk_set_reports_unknown_empty_and_unparsed_corpora(
    isolated_database: Path,
) -> None:
    """Verify invalid corpus states produce distinct typed errors.

    Args:
        isolated_database: Initialized temporary application database.

    Returns:
        None. Assertions cover all pre-computation validation branches.
    """
    tokenizer = CountingTokenizer()

    # An identifier absent from SQLite is different from an existing empty corpus.
    with pytest.raises(ChunkingCorpusNotFoundError):
        build_or_reuse_chunk_set("missing", ChunkingConfig(), tokenizer)

    _insert_corpus()
    with pytest.raises(EmptyChunkingCorpusError):
        build_or_reuse_chunk_set("corpus-1", ChunkingConfig(), tokenizer)

    # A legacy document row without a canonical parse blocks the complete artifact.
    with connect() as connection:
        connection.execute(
            "INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "document-missing",
                "corpus-1",
                "missing.txt",
                "uploads/missing.txt",
                "text/plain",
                1,
                "0" * 64,
                "2026-08-14T00:00:01Z",
            ),
        )

    with pytest.raises(MissingParseArtifactError) as error:
        build_or_reuse_chunk_set("corpus-1", ChunkingConfig(), tokenizer)
    assert error.value.document_ids == ["document-missing"]


def test_chunk_set_persistence_rolls_back_complete_artifact(
    isolated_database: Path,
) -> None:
    """Verify a child insert failure leaves no parent or partial chunks.

    Args:
        isolated_database: Initialized temporary application database.

    Returns:
        None. Assertions inspect SQLite after the simulated failure.
    """
    _seed_complete_corpus()

    # Fail at the repository's child-write boundary inside the parent transaction.
    with (
        patch.object(
            chunk_set_repository,
            "_insert_chunk",
            side_effect=sqlite3.OperationalError("simulated write failure"),
        ),
        pytest.raises(sqlite3.OperationalError, match="simulated"),
    ):
        build_or_reuse_chunk_set(
            "corpus-1",
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size_tokens=2,
                chunk_overlap_tokens=0,
            ),
            CountingTokenizer(),
        )

    # Transaction rollback removes the parent row as well as all potential children.
    with sqlite3.connect(isolated_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunk_set").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM chunk").fetchone()[0] == 0
