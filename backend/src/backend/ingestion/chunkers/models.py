"""Shared immutable contracts for tokenization and document chunking."""

from dataclasses import dataclass
from typing import Protocol

from backend.pipeline.configs import ChunkingConfig


@dataclass(frozen=True)
class TokenOffset:
    """Locate one tokenizer token in its original input text.

    Attributes:
        start: Zero-based, inclusive Unicode character offset.
        end: Zero-based, exclusive Unicode character offset.
    """

    start: int
    end: int


@dataclass(frozen=True)
class TokenizedText:
    """Store the source offsets returned by one tokenizer encoding.

    Attributes:
        offsets: Ordered source-text offsets for all non-special tokens.
    """

    offsets: tuple[TokenOffset, ...]


@dataclass(frozen=True)
class ChunkSpan:
    """Describe one exact slice of canonical document text.

    Attributes:
        text: Exact canonical-text slice persisted as the chunk body.
        character_start_offset: Inclusive document character offset.
        character_end_offset: Exclusive document character offset.
        token_start_offset: Inclusive document token offset when exact.
        token_end_offset: Exclusive document token offset when exact.
    """

    text: str
    character_start_offset: int
    character_end_offset: int
    token_start_offset: int | None
    token_end_offset: int | None


class ChunkingTokenizer(Protocol):
    """Define the provider-neutral tokenizer boundary needed by chunkers."""

    identifier: str
    revision: str
    asset_sha256: str
    special_tokens_policy: str

    def encode(self, text: str) -> TokenizedText:
        """Tokenize text and return offsets into the submitted string.

        Args:
            text: Unicode source text to measure.

        Returns:
            Ordered non-special-token source offsets.
        """
        ...


class Chunker(Protocol):
    """Define the common behavior implemented by every chunking strategy."""

    name: str
    version: str

    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
        tokenizer: ChunkingTokenizer,
    ) -> list[ChunkSpan]:
        """Split canonical text under the supplied resolved configuration.

        Args:
            text: Complete canonical document text.
            config: Validated and default-resolved chunking configuration.
            tokenizer: Token-counting and offset adapter.

        Returns:
            Ordered exact slices of the canonical document.
        """
        ...
