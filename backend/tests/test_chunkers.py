"""Unit tests for offline tokenization and all three chunking strategies."""

import re
from pathlib import Path

import pytest

from backend.ingestion.chunkers.models import TokenizedText, TokenOffset
from backend.ingestion.chunkers.strategies import (
    MAX_CHUNK_CHARACTERS,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
)
from backend.ingestion.chunkers.tokenizer import (
    TOKENIZER_PATH,
    MultilingualBertTokenizer,
    TokenizerAssetError,
)
from backend.pipeline.configs import ChunkingConfig, ChunkingStrategy


class WhitespaceTokenizer:
    """Provide deterministic word-like offsets without external model work."""

    identifier = "test-whitespace"
    revision = "1"
    asset_sha256 = "test-digest"
    special_tokens_policy = "none"

    def encode(self, text: str) -> TokenizedText:
        """Return offsets for each non-whitespace sequence.

        Args:
            text: Source string measured by the fake tokenizer.

        Returns:
            Ordered offsets for regex-delimited test tokens.
        """
        # Model a small tokenizer while retaining exact source offset behavior.
        return TokenizedText(
            offsets=tuple(
                TokenOffset(match.start(), match.end())
                for match in re.finditer(r"\S+", text)
            )
        )


class SingleTokenTokenizer(WhitespaceTokenizer):
    """Model a pathological tokenizer that maps all input to one unknown token."""

    def encode(self, text: str) -> TokenizedText:
        """Return one token spanning the entire non-empty input.

        Args:
            text: Source string represented by the unknown token.

        Returns:
            One full-input offset, or no offsets for empty input.
        """
        # Reproduce WordPiece's possible long unknown-token behavior.
        if not text:
            return TokenizedText(offsets=())
        return TokenizedText(offsets=(TokenOffset(0, len(text)),))


class CharacterTokenizer(WhitespaceTokenizer):
    """Provide one test token per non-whitespace Unicode character."""

    def encode(self, text: str) -> TokenizedText:
        """Return character-sized offsets for multilingual boundary tests.

        Args:
            text: Unicode source text measured by the fake tokenizer.

        Returns:
            One offset for each non-whitespace Python character.
        """
        # Character tokens make punctuation boundaries visible without model behavior.
        return TokenizedText(
            offsets=tuple(
                TokenOffset(index, index + 1)
                for index, character in enumerate(text)
                if not character.isspace()
            )
        )


def test_local_tokenizer_loads_offline_without_special_tokens() -> None:
    """Verify the pinned adapter returns exact Unicode source offsets.

    Args:
        None.

    Returns:
        None. Assertions cover local loading, offsets, and special-token policy.
    """
    tokenizer = MultilingualBertTokenizer()
    text = "Hello नमस्ते 世界"
    encoding = tokenizer.encode(text)

    # Every reported token must resolve directly into the original Unicode string.
    assert encoding.offsets
    assert all(text[offset.start : offset.end] for offset in encoding.offsets)
    assert encoding.offsets[0] == TokenOffset(0, 5)

    # No model-only CLS or SEP entries should consume the chunk budget.
    assert all(offset != TokenOffset(0, 0) for offset in encoding.offsets)


def test_local_tokenizer_rejects_unpinned_digest() -> None:
    """Verify a modified tokenizer identity fails before runtime use.

    Args:
        None.

    Returns:
        None. A stable domain error confirms digest enforcement.
    """
    # Supply an intentionally incorrect expected digest against the real local asset.
    with pytest.raises(TokenizerAssetError, match="pinned SHA-256"):
        MultilingualBertTokenizer(
            asset_path=Path(TOKENIZER_PATH),
            expected_sha256="0" * 64,
        )


def test_fixed_size_chunker_uses_exact_overlap_and_no_trailing_window() -> None:
    """Verify fixed windows overlap precisely and stop at the final token.

    Args:
        None.

    Returns:
        None. Assertions verify text and global token intervals.
    """
    chunks = FixedSizeChunker().chunk(
        "one two three four five",
        ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size_tokens=3,
            chunk_overlap_tokens=1,
        ),
        WhitespaceTokenizer(),
    )

    # The second window shares exactly token two of the first zero-based interval.
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
    assert [(chunk.token_start_offset, chunk.token_end_offset) for chunk in chunks] == [
        (0, 3),
        (2, 5),
    ]


def test_recursive_chunker_prefers_boundaries_and_keeps_exact_overlap() -> None:
    """Verify recursive chunks end naturally and preserve configured overlap.

    Args:
        None.

    Returns:
        None. Assertions cover sentence fallback and exact token reuse.
    """
    text = "One two. Three four five. Six seven eight."
    chunks = RecursiveChunker().chunk(
        text,
        ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size_tokens=5,
            chunk_overlap_tokens=1,
        ),
        WhitespaceTokenizer(),
    )

    # Oversized text falls back to sentences before using arbitrary token endings.
    assert chunks[0].text == "One two. Three four five."
    assert chunks[1].text == "five. Six seven eight."
    assert chunks[1].token_start_offset == chunks[0].token_end_offset - 1


def test_recursive_chunker_recognizes_unspaced_multilingual_sentences() -> None:
    """Verify CJK sentence punctuation works without following whitespace.

    Args:
        None.

    Returns:
        None. Assertions cover the multilingual sentence-boundary fallback.
    """
    chunks = RecursiveChunker().chunk(
        "甲乙。丙丁。戊己。",
        ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size_tokens=5,
            chunk_overlap_tokens=1,
        ),
        CharacterTokenizer(),
    )

    # The first window ends after a complete sentence rather than token five.
    assert chunks[0].text == "甲乙。"
    assert chunks[1].token_start_offset == chunks[0].token_end_offset - 1


def test_paragraph_chunker_packs_complete_units_and_splits_oversized_unit() -> None:
    """Verify blank lines delimit paragraphs while single newlines remain content.

    Args:
        None.

    Returns:
        None. Assertions cover packing, separators, and no-overlap fallback.
    """
    text = "One\nline\n\nTwo words\n\nThree four five six seven"
    chunks = ParagraphChunker().chunk(
        text,
        ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size_tokens=4,
        ),
        WhitespaceTokenizer(),
    )

    # The first two paragraphs fit together and preserve their original separators.
    assert [chunk.text for chunk in chunks] == [
        "One\nline\n\nTwo words",
        "Three four five six",
        "seven",
    ]
    assert [(chunk.token_start_offset, chunk.token_end_offset) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 9),
    ]


def test_character_cap_handles_one_pathological_token() -> None:
    """Verify one enormous unknown token is split at safe character boundaries.

    Args:
        None.

    Returns:
        None. Assertions verify bounded progress and unavailable token provenance.
    """
    text = "x" * (MAX_CHUNK_CHARACTERS + 7)
    chunks = FixedSizeChunker().chunk(
        text,
        ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size_tokens=1,
            chunk_overlap_tokens=0,
        ),
        SingleTokenTokenizer(),
    )

    # Character fallback is the only case where exact global token offsets are omitted.
    assert [len(chunk.text) for chunk in chunks] == [MAX_CHUNK_CHARACTERS, 7]
    assert all(chunk.token_start_offset is None for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


@pytest.mark.parametrize(
    ("chunker", "config"),
    [
        (
            FixedSizeChunker(),
            ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size_tokens=3,
                chunk_overlap_tokens=1,
            ),
        ),
        (
            RecursiveChunker(),
            ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size_tokens=3,
                chunk_overlap_tokens=1,
            ),
        ),
        (
            ParagraphChunker(),
            ChunkingConfig(
                strategy=ChunkingStrategy.PARAGRAPH,
                chunk_size_tokens=3,
            ),
        ),
    ],
)
def test_all_chunkers_emit_exact_non_empty_bounded_slices(
    chunker: object,
    config: ChunkingConfig,
) -> None:
    """Verify shared source-slice and hard-limit invariants.

    Args:
        chunker: Stateless strategy implementation under test.
        config: Matching resolved strategy configuration.

    Returns:
        None. Assertions cover common chunk correctness invariants.
    """
    text = "alpha beta\n\ngamma delta epsilon"
    chunks = chunker.chunk(text, config, WhitespaceTokenizer())

    # Every persisted body must be recoverable from its canonical source offsets.
    for chunk in chunks:
        assert chunk.text
        assert (
            chunk.text
            == text[chunk.character_start_offset : chunk.character_end_offset]
        )
        assert len(chunk.text) <= MAX_CHUNK_CHARACTERS
        if chunk.token_start_offset is not None:
            assert chunk.token_end_offset - chunk.token_start_offset <= 3
