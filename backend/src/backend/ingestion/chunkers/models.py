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


@dataclass(frozen=True, init=False)
class TokenizedText:
    """Store reusable source-boundary arrays returned by one tokenizer encoding.

    Attributes:
        token_starts: Ordered inclusive character starts for non-special tokens.
        token_ends: Ordered exclusive character ends for non-special tokens.

    ``offsets`` remains available as a compatibility view for injected tokenizers and
    existing callers. Production chunking uses the parallel integer tuples directly,
    avoiding one long-lived Python dataclass allocation for every document token.
    """

    token_starts: tuple[int, ...]
    token_ends: tuple[int, ...]

    def __init__(
        self,
        offsets: tuple[TokenOffset, ...] | None = None,
        *,
        token_starts: tuple[int, ...] | None = None,
        token_ends: tuple[int, ...] | None = None,
    ) -> None:
        """Create token boundaries from legacy offsets or compact parallel arrays.

        Args:
            offsets: Optional legacy token-offset objects supplied by existing adapters.
            token_starts: Optional compact inclusive character-start tuple.
            token_ends: Optional compact exclusive character-end tuple.

        Returns:
            None. The frozen instance stores one validated pair of boundary tuples.

        Raises:
            ValueError: If both representations are supplied or lengths do not match.
        """
        # Legacy injected tokenizers still provide TokenOffset objects. Convert that
        # representation once so all chunking helpers receive the same compact arrays.
        if offsets is not None:
            if token_starts is not None or token_ends is not None:
                raise ValueError(
                    "Provide offsets or token_starts/token_ends, not both."
                )
            resolved_starts = tuple(offset.start for offset in offsets)
            resolved_ends = tuple(offset.end for offset in offsets)
        else:
            # Production tokenizers provide parallel arrays and avoid constructing a
            # TokenOffset object for each token in a potentially very large document.
            resolved_starts = token_starts or ()
            resolved_ends = token_ends or ()

        # Parallel tuples must describe the same number of tokens so any shared index
        # safely addresses both the inclusive start and exclusive end of one token.
        if len(resolved_starts) != len(resolved_ends):
            raise ValueError("Token start and end arrays must have equal lengths.")

        object.__setattr__(self, "token_starts", resolved_starts)
        object.__setattr__(self, "token_ends", resolved_ends)

    @property
    def offsets(self) -> tuple[TokenOffset, ...]:
        """Materialize the legacy object view only when a caller explicitly needs it.

        Args:
            None.

        Returns:
            Ordered TokenOffset objects corresponding to the compact boundary arrays.
        """
        # zip pairs matching positions; strict=True is safe after constructor validation.
        return tuple(
            TokenOffset(start=start, end=end)
            for start, end in zip(self.token_starts, self.token_ends, strict=True)
        )


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
        """Tokenize text and return reusable boundaries into the submitted string.

        Args:
            text: Unicode source text to measure.

        Returns:
            Ordered non-special-token source boundary arrays.
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
