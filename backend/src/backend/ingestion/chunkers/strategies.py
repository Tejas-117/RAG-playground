"""Split canonical document text with three token-aware chunking strategies.

All strategies work with two related coordinate systems:

* Token indices address matching entries in the tokenizer's ``token_starts`` and
  ``token_ends`` tuples. Token ranges are half-open, so ``[0, 4)`` contains tokens
  0 through 3.
* Character offsets identify positions in the persisted canonical text. They are
  also half-open, so ``text[character_start:character_end]`` reproduces a chunk.

The tokenizer supplies the mapping between these coordinate systems. Strategies
choose boundaries in token space to honor token budgets, then ``_span_from_tokens``
uses the mapped character offsets to retain the exact source text and provenance.
"""

import re
from bisect import bisect_left, bisect_right

from backend.ingestion.chunkers.models import (
    Chunker,
    ChunkingTokenizer,
    ChunkSpan,
    TokenizedText,
)
from backend.pipeline.configs import ChunkingConfig, ChunkingStrategy

# Limit characters separately from tokens. A tokenizer can represent an extremely
# long URL or generated identifier as one unknown token, so a token-only limit would
# still permit an impractically large chunk.
MAX_CHUNK_CHARACTERS = 32_000

# Include the boundary-algorithm version in chunk-set fingerprints. Incrementing this
# value prevents reuse of chunks made by an older implementation with different output.
CHUNKER_VERSION = "1.0.0"

# Recursive chunking attempts these boundaries from the largest semantic unit to the
# smallest. It only proceeds to a finer separator when the current unit is too large.
_RECURSIVE_SEPARATORS = (
    # Divide at blank lines, including blank lines containing horizontal whitespace.
    re.compile(r"\n[^\S\n]*\n+"),
    # Divide after common Latin or CJK sentence-ending punctuation.
    re.compile(r"(?<=[.!?。！？])(?:[ \t]+|\n+|(?=\S))"),
    # Divide at any remaining single or repeated newline.
    re.compile(r"\n+"),
    # Divide at remaining whitespace between words as the finest natural boundary.
    re.compile(r"\s+"),
)

# Reuse the blank-line rule as the only separator for paragraph chunking.
_PARAGRAPH_SEPARATOR = _RECURSIVE_SEPARATORS[0]


class FixedSizeChunker:
    """Create uniform token windows and repeat the configured overlap.

    This strategy ignores paragraphs and sentences. For a chunk size of four and
    overlap of one, it emits token ranges ``[0, 4)``, ``[3, 7)``, ``[6, 10)``, and
    so on. Fixed windows make chunk sizes predictable and are useful as a baseline.
    """

    name = ChunkingStrategy.FIXED_SIZE.value
    version = CHUNKER_VERSION

    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
        tokenizer: ChunkingTokenizer,
    ) -> list[ChunkSpan]:
        """Split canonical text into fixed-width, overlapping token windows.

        The document is tokenized once. ``token_start`` is an index in the returned
        boundary arrays, while ``chunk_size_tokens`` is a count. Adding them produces
        the exclusive token boundary for a window, just like a Python slice. The
        stride subtracts overlap so the requested trailing tokens are repeated at
        the beginning of the next chunk.

        Args:
            text: Complete canonical document text.
            config: Fixed-size chunk limits and overlap.
            tokenizer: Adapter returning global source offsets.

        Returns:
            Ordered exact canonical-text slices with document-global offsets.
        """
        # Parallel boundary arrays map every global token index to its exact source
        # character interval without allocating an object for every document token.
        tokenized = tokenizer.encode(text)

        # Whitespace-only or unusual input may produce no tokenizer-visible tokens.
        # Character fallback still makes bounded progress for any non-empty content.
        if not tokenized.token_starts:
            return _character_fallback(text, 0, len(text))

        # Configuration validation guarantees overlap is smaller than chunk size.
        # The stride is the number of new tokens consumed after each window; for
        # size=4 and overlap=1, the next chunk begins three token positions later.
        overlap = config.chunk_overlap_tokens or 0
        stride = config.chunk_size_tokens - overlap
        spans: list[ChunkSpan] = []
        token_start = 0

        # Build half-open token windows [token_start, token_end). token_end is an
        # exclusive boundary rather than the index of the last included token.
        while token_start < len(tokenized.token_starts):
            # Cap the calculated end at the total token count so the last window can
            # be shorter than chunk_size_tokens without indexing beyond offsets.
            token_end = min(
                token_start + config.chunk_size_tokens,
                len(tokenized.token_starts),
            )

            # Convert token indices into an exact character slice. The independent
            # character guard handles pathological cases such as one enormous token.
            span = _span_from_tokens(text, tokenized, token_start, token_end)
            spans.extend(_enforce_character_limit(text, span))

            # Once this window includes the last token, another stride would create
            # a redundant chunk containing only tokens already used as overlap.
            if token_end == len(tokenized.token_starts):
                break

            # Move past the newly consumed tokens while retaining the configured
            # number of trailing tokens at the start of the next window.
            token_start += stride

        return spans


class RecursiveChunker:
    """Prefer structural text boundaries while enforcing a token budget.

    Recursive chunking first finds usable ends at paragraph, sentence, line, or
    whitespace boundaries. It then builds windows ending at the furthest discovered
    boundary that fits. Oversized units eventually fall back to raw token boundaries.
    """

    name = ChunkingStrategy.RECURSIVE.value
    version = CHUNKER_VERSION

    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
        tokenizer: ChunkingTokenizer,
    ) -> list[ChunkSpan]:
        """Discover natural boundaries, then emit overlapped token windows.

        Boundary discovery and window construction are separate phases. Discovery
        recursively subdivides only units that exceed the token or character limit.
        Construction chooses the furthest discovered end inside each allowed window,
        converts that token range to source text, and moves the next start backward
        by the configured overlap.

        Args:
            text: Complete canonical document text.
            config: Recursive chunk limits and overlap.
            tokenizer: Adapter returning global source offsets.

        Returns:
            Ordered canonical-text slices that prefer natural ending boundaries.
        """
        # Parallel boundary arrays enforce token counts while retaining exact source
        # slices, and every recursive helper reuses these same arrays.
        tokenized = tokenizer.encode(text)

        # If no token boundaries exist, split directly by characters so unusual input
        # cannot disappear or cause the token-window loop to stall.
        if not tokenized.token_starts:
            return _character_fallback(text, 0, len(text))

        # Phase 1: recursively inspect the entire token range and collect global,
        # exclusive token indices that represent acceptable natural unit endings.
        natural_ends = tuple(
            sorted(
                _collect_recursive_leaf_ends(
                    text,
                    tokenized,
                    0,
                    len(tokenized.token_starts),
                    config.chunk_size_tokens,
                    0,
                )
            )
        )
        overlap = config.chunk_overlap_tokens or 0
        spans: list[ChunkSpan] = []
        token_start = 0

        # Phase 2: construct chunks from the discovered ends. token_start remains a
        # document-global token index even when overlap moves it into a prior unit.
        while token_start < len(tokenized.token_starts):
            # The token budget establishes the furthest possible end. The character
            # safety limit may pull it earlier when normal tokens cover excessive text.
            maximum_end = min(
                token_start + config.chunk_size_tokens,
                len(tokenized.token_starts),
            )
            maximum_end = _fit_character_limit(
                tokenized,
                token_start,
                maximum_end,
            )

            # Binary-search the sorted natural boundaries for the rightmost end within
            # the current hard limits instead of scanning every document boundary.
            candidate_index = bisect_right(natural_ends, maximum_end) - 1
            token_end = (
                natural_ends[candidate_index]
                if candidate_index >= 0 and natural_ends[candidate_index] > token_start
                else maximum_end
            )

            # A selected unit can be no larger than the requested overlap. In that
            # case token_end - overlap would fail to advance, so use the full allowed
            # window when possible and guarantee forward movement below.
            if token_end <= token_start + overlap and maximum_end > token_end:
                token_end = maximum_end

            # Materialize the selected global token interval as persisted source text.
            span = _span_from_tokens(text, tokenized, token_start, token_end)
            spans.extend(_enforce_character_limit(text, span))

            # Do not emit a final overlap-only window after reaching the document end.
            if token_end == len(tokenized.token_starts):
                break

            # Normally the next start repeats exactly overlap tokens. max() also
            # guarantees at least one-token progress for exceptionally short chunks.
            token_start = max(token_start + 1, token_end - overlap)

        return spans


class ParagraphChunker:
    """Keep blank-line-delimited paragraphs whole whenever limits permit.

    Adjacent paragraphs are greedily combined to avoid many tiny chunks. Paragraphs
    are never overlapped. Only a paragraph that cannot fit by itself is divided into
    consecutive token windows, because no paragraph-preserving solution exists.
    """

    name = ChunkingStrategy.PARAGRAPH.value
    version = CHUNKER_VERSION

    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
        tokenizer: ChunkingTokenizer,
    ) -> list[ChunkSpan]:
        """Pack complete paragraphs and split only an oversized paragraph.

        Blank-line separators identify paragraph token ranges. The method accumulates
        adjacent ranges while their combined token and character sizes fit. Before an
        oversized paragraph is split, any previously accumulated paragraphs are
        flushed so chunks remain ordered and never mix partial paragraphs.

        Args:
            text: Complete canonical document text.
            config: Paragraph token limit with resolved zero overlap.
            tokenizer: Adapter returning global source offsets.

        Returns:
            Ordered, non-overlapping canonical-text slices.
        """
        # Tokenize once so paragraph character ranges can be mapped through shared,
        # document-global boundary arrays without repeatedly rebuilding those arrays.
        tokenized = tokenizer.encode(text)

        # Preserve bounded content through character slicing if the tokenizer exposes
        # no usable offsets for this document.
        if not tokenized.token_starts:
            return _character_fallback(text, 0, len(text))

        # Convert every content-bearing, blank-line-delimited character region into a
        # half-open global token range such as [paragraph_start, paragraph_end).
        paragraph_ranges = _token_ranges_for_split_text(
            text,
            tokenized,
            0,
            len(text),
            _PARAGRAPH_SEPARATOR,
        )
        spans: list[ChunkSpan] = []
        packed_start: int | None = None
        packed_end: int | None = None

        # Walk paragraphs in document order. packed_start/packed_end describe the
        # token range of complete paragraphs accumulated for the next output chunk.
        for paragraph_start, paragraph_end in paragraph_ranges:
            # Token ranges are half-open, so subtraction gives the exact token count.
            # Character size comes from the first included token start through the
            # final included token end and therefore includes internal separators.
            paragraph_tokens = paragraph_end - paragraph_start
            paragraph_characters = (
                tokenized.token_ends[paragraph_end - 1]
                - tokenized.token_starts[paragraph_start]
            )

            # A paragraph exceeding either hard limit cannot remain whole. First emit
            # any prior complete-paragraph pack, then split only this paragraph into
            # consecutive windows so output ordering remains unchanged.
            if (
                paragraph_tokens > config.chunk_size_tokens
                or paragraph_characters > MAX_CHUNK_CHARACTERS
            ):
                if packed_start is not None and packed_end is not None:
                    spans.append(
                        _span_from_tokens(text, tokenized, packed_start, packed_end)
                    )
                    packed_start = None
                    packed_end = None

                spans.extend(
                    _non_overlapping_token_spans(
                        text,
                        tokenized,
                        paragraph_start,
                        paragraph_end,
                        config.chunk_size_tokens,
                    )
                )
                continue

            # With no active pack, make this complete paragraph the first candidate.
            if packed_start is None:
                packed_start = paragraph_start
                packed_end = paragraph_end
                continue

            # Measure the hypothetical range from the first packed paragraph through
            # the current paragraph, including whitespace separating those paragraphs.
            combined_tokens = paragraph_end - packed_start
            combined_characters = (
                tokenized.token_ends[paragraph_end - 1]
                - tokenized.token_starts[packed_start]
            )

            # If the combined range fits, retain it for possible further packing.
            # Otherwise emit the previous pack and begin a new one at this paragraph.
            if (
                combined_tokens <= config.chunk_size_tokens
                and combined_characters <= MAX_CHUNK_CHARACTERS
            ):
                packed_end = paragraph_end
            else:
                spans.append(
                    _span_from_tokens(text, tokenized, packed_start, packed_end)
                )
                packed_start = paragraph_start
                packed_end = paragraph_end

        # The loop emits a pack only when the next paragraph does not fit. Explicitly
        # emit the final accumulated pack after there is no next paragraph to trigger it.
        if packed_start is not None and packed_end is not None:
            spans.append(_span_from_tokens(text, tokenized, packed_start, packed_end))

        return spans


def get_chunker(strategy: ChunkingStrategy) -> Chunker:
    """Map a validated configuration strategy to its implementation.

    Chunk-set orchestration depends on the small ``Chunker`` protocol rather than a
    concrete strategy. Keeping the mapping here lets callers select an implementation
    without importing strategy-specific classes or adding selection logic elsewhere.

    Args:
        strategy: Validated chunking strategy identifier.

    Returns:
        A new stateless chunker implementing the shared ``Chunker`` protocol.
    """
    # Construct every supported stateless implementation in one exhaustive mapping.
    # Dictionary lookup also makes an unsupported enum value fail immediately.
    implementations: dict[ChunkingStrategy, Chunker] = {
        ChunkingStrategy.RECURSIVE: RecursiveChunker(),
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker(),
        ChunkingStrategy.PARAGRAPH: ParagraphChunker(),
    }
    return implementations[strategy]


def _span_from_tokens(
    text: str,
    tokenized: TokenizedText,
    token_start: int,
    token_end: int,
) -> ChunkSpan:
    """Translate a half-open token interval into source text and provenance.

    A token interval does not directly contain source text. Its first token supplies
    the inclusive character start and its final included token supplies the exclusive
    character end. Slicing canonical text with those boundaries preserves the exact
    source instead of approximately reconstructing it from tokenizer IDs.

    Args:
        text: Complete canonical document text.
        tokenized: Reusable start/end character arrays for all document tokens.
        token_start: Inclusive global token index.
        token_end: Exclusive global token index.

    Returns:
        Exact text with document-global token and character boundaries.
    """
    # token_end is exclusive, so token_end - 1 is the final token included in the
    # chunk. The resulting character values use the same half-open convention.
    character_start = tokenized.token_starts[token_start]
    character_end = tokenized.token_ends[token_end - 1]

    # Retain both coordinate systems so persistence and citation code can trace this
    # exact canonical slice without tokenizing the document again.
    return ChunkSpan(
        text=text[character_start:character_end],
        character_start_offset=character_start,
        character_end_offset=character_end,
        token_start_offset=token_start,
        token_end_offset=token_end,
    )


def _non_overlapping_token_spans(
    text: str,
    tokenized: TokenizedText,
    token_start: int,
    token_end: int,
    chunk_size: int,
) -> list[ChunkSpan]:
    """Split one oversized structural unit into consecutive token windows.

    Paragraph chunking calls this helper when one paragraph cannot fit whole. Each
    window begins exactly where the previous one ended, preserving order without
    introducing overlap that the paragraph strategy promises not to use.

    Args:
        text: Complete canonical document text.
        tokenized: Reusable start/end character arrays for all document tokens.
        token_start: Inclusive token index of the oversized unit.
        token_end: Exclusive token index of the oversized unit.
        chunk_size: Maximum tokens in each emitted window.

    Returns:
        Ordered, non-overlapping spans covering the supplied token interval.
    """
    spans: list[ChunkSpan] = []
    cursor = token_start

    # Consume [cursor, window_end), then begin exactly at window_end. No token is
    # repeated between adjacent windows because paragraph chunks do not overlap.
    while cursor < token_end:
        # The final window may be shorter than chunk_size, so cap it at the exclusive
        # end of this paragraph rather than the end of the complete document.
        window_end = min(cursor + chunk_size, token_end)

        # Convert the token window to source text, then apply the separate safeguard
        # for an abnormally long tokenizer token.
        span = _span_from_tokens(text, tokenized, cursor, window_end)
        spans.extend(_enforce_character_limit(text, span))
        cursor = window_end

    return spans


def _enforce_character_limit(text: str, span: ChunkSpan) -> list[ChunkSpan]:
    """Apply the independent character safety cap to a candidate chunk.

    Token budgets normally keep chunks small, but WordPiece can represent a very long
    unknown string as one token. If that makes a span exceed the character cap, this
    helper replaces it with bounded character slices. Those fallback slices cannot
    claim exact token boundaries because they may divide one tokenizer token.

    Args:
        text: Complete canonical document text.
        span: Candidate span already within its token budget.

    Returns:
        The unchanged safe span, or ordered character-bounded replacements.
    """
    # Keep the original span and its precise token provenance when it is already safe.
    if len(span.text) <= MAX_CHUNK_CHARACTERS:
        return [span]

    return _character_fallback(
        text,
        span.character_start_offset,
        span.character_end_offset,
    )


def _character_fallback(text: str, start: int, end: int) -> list[ChunkSpan]:
    """Split a character interval when token boundaries cannot safely be used.

    Python string slicing operates on Unicode code points, so this fallback avoids
    cutting encoded bytes in the middle of a character. Token offsets are set to
    ``None`` because the tokenizer either emitted no offsets or a giant token had to
    be divided, making precise token provenance unavailable.

    Args:
        text: Complete canonical document text.
        start: Inclusive fallback character offset.
        end: Exclusive fallback character offset.

    Returns:
        Content-bearing exact slices with unavailable token offsets set to ``None``.
    """
    spans: list[ChunkSpan] = []
    cursor = start

    # Advance through the supplied half-open interval in fixed character windows.
    # min() permits the final window to be shorter than the safety limit.
    while cursor < end:
        window_end = min(cursor + MAX_CHUNK_CHARACTERS, end)
        window = text[cursor:window_end]

        # Whitespace-only windows have no retrievable content and are not persisted.
        # Every retained window still keeps its original document character offsets.
        if window.strip():
            spans.append(
                ChunkSpan(
                    text=window,
                    character_start_offset=cursor,
                    character_end_offset=window_end,
                    token_start_offset=None,
                    token_end_offset=None,
                )
            )

        cursor = window_end

    return spans


def _fit_character_limit(
    tokenized: TokenizedText,
    token_start: int,
    maximum_end: int,
) -> int:
    """Pull a token-limited window end back to the character safety limit.

    Token offsets are ordered by character end. ``bisect_right`` finds the exclusive
    boundary after the last token ending at or before the allowed character position.
    At least one token is returned so callers always advance; if that token is itself
    too long, ``_enforce_character_limit`` later divides it by characters.

    Args:
        tokenized: Reusable start/end character arrays for all document tokens.
        token_start: Inclusive candidate token start.
        maximum_end: Exclusive end imposed by the token limit.

    Returns:
        Exclusive token end that advances by at least one token.
    """
    # Convert the relative character budget into an absolute document position.
    character_limit = tokenized.token_starts[token_start] + MAX_CHUNK_CHARACTERS

    # Search only within the current token-limited window so the fitted boundary can
    # never exceed maximum_end even if later tokens also fit the character position.
    fitted_end = bisect_right(
        tokenized.token_ends,
        character_limit,
        token_start,
        maximum_end,
    )

    # Force one-token progress when no whole token fits under the character limit.
    return max(token_start + 1, fitted_end)


def _collect_recursive_leaf_ends(
    text: str,
    tokenized: TokenizedText,
    token_start: int,
    token_end: int,
    chunk_size: int,
    separator_level: int,
) -> set[int]:
    """Find usable natural ends by subdividing only oversized text units.

    The initial call represents the whole document and starts with the paragraph
    separator. If that unit is too large, it is divided into paragraphs and each
    paragraph is checked independently at the next separator level. A paragraph that
    already fits contributes its exclusive token end immediately; an oversized one is
    split further into sentences, lines, whitespace units, and finally raw tokens.

    The returned set contains global token boundaries, not ready-made chunks. The
    recursive chunker later chooses among these boundaries while applying overlap.

    TL;DR of the cooperating helpers:
        - ``_collect_recursive_leaf_ends`` keeps subdividing an oversized text unit
          until every resulting unit fits within the configured limits, then records
          each unit's exclusive ending token index.
        - ``_split_character_ranges`` divides a requested character interval wherever
          the current separator matches and returns the content between separators.
        - ``_token_range_for_characters`` maps one resulting character interval back
          to the global token-index interval containing its intersecting tokens.

    Flow:
        Oversized token range
        -> convert it to its canonical character range
        -> split that character range with the current separator
        -> map each character piece back to a global token range
        -> recursively process every piece that is still oversized

    Args:
        text: Complete canonical document text.
        tokenized: Reusable start/end character arrays for all document tokens.
        token_start: Inclusive token index of the current unit.
        token_end: Exclusive token index of the current unit.
        chunk_size: Maximum configured token count.
        separator_level: Index of the separator currently being attempted.

    Returns:
        Exclusive global token indices usable as preferred chunk endings.
    """
    # Calculate the source-character width covered by this token unit. Both the token
    # count and the character width must fit before the unit can remain whole.
    characters = (
        tokenized.token_ends[token_end - 1] - tokenized.token_starts[token_start]
    )

    # Stop descending as soon as the current structural unit fits. For example, a
    # fitting paragraph remains one leaf instead of being needlessly split by sentence.
    if token_end - token_start <= chunk_size and characters <= MAX_CHUNK_CHARACTERS:
        return {token_end}

    # Once every natural separator has been tried, manufacture boundaries every
    # chunk_size tokens. Including token_end ensures the remaining tail is retained.
    if separator_level == len(_RECURSIVE_SEPARATORS):
        return set(range(token_start + chunk_size, token_end, chunk_size)) | {token_end}

    # Translate this token unit back to the exact canonical character region because
    # regular-expression separators operate on text rather than tokenizer indices.
    character_start = tokenized.token_starts[token_start]
    character_end = tokenized.token_ends[token_end - 1]

    # Split with the current structural rule and map every content-bearing result back
    # into document-global token indices for the next recursive step.
    ranges = _token_ranges_for_split_text(
        text,
        tokenized,
        character_start,
        character_end,
        _RECURSIVE_SEPARATORS[separator_level],
    )

    # Zero or one result means this separator did not meaningfully divide the unit.
    # Retry the same token range with the next finer separator instead.
    if len(ranges) <= 1:
        return _collect_recursive_leaf_ends(
            text,
            tokenized,
            token_start,
            token_end,
            chunk_size,
            separator_level + 1,
        )

    leaf_ends: set[int] = set()

    # Process every child independently. Small children stop immediately, while only
    # oversized children descend to finer boundaries. Union their global end indices.
    for child_start, child_end in ranges:
        leaf_ends.update(
            _collect_recursive_leaf_ends(
                text,
                tokenized,
                child_start,
                child_end,
                chunk_size,
                separator_level + 1,
            )
        )

    return leaf_ends


def _token_ranges_for_split_text(
    text: str,
    tokenized: TokenizedText,
    character_start: int,
    character_end: int,
    separator: re.Pattern[str],
) -> list[tuple[int, int]]:
    """Apply one text separator and map its pieces into global token space.

    Regular expressions locate boundaries using canonical character positions, while
    chunk-size checks use token indices. This adapter performs both steps: it finds
    content regions between separator matches and converts every region to the tokens
    that intersect it. Empty or tokenizer-invisible regions are omitted.

    Args:
        text: Complete canonical document text.
        tokenized: Reusable start/end character arrays for all document tokens.
        character_start: Inclusive region character offset.
        character_end: Exclusive region character offset.
        separator: Compiled expression identifying boundaries between text units.

    Returns:
        Ordered half-open global token intervals containing visible tokens.
    """
    # First find the candidate pieces in the coordinate system used by the regex.
    character_ranges = _split_character_ranges(
        text,
        character_start,
        character_end,
        separator,
    )
    token_ranges: list[tuple[int, int]] = []

    # Then map every character piece to the token-index coordinate system used by the
    # chunking algorithms. The indices remain global to the complete document.
    for start, end in character_ranges:
        token_range = _token_range_for_characters(tokenized, start, end)

        # A half-open interval is non-empty only when start < end. Ignore separators or
        # unusual regions that contain no tokenizer-visible token.
        if token_range[0] < token_range[1]:
            token_ranges.append(token_range)

    return token_ranges


def _split_character_ranges(
    text: str,
    start: int,
    end: int,
    separator: re.Pattern[str],
) -> list[tuple[int, int]]:
    """Return content regions found between matches of one separator rule.

    Separator text itself is excluded from the individual ranges. When several
    adjacent ranges are later packed into one chunk, slicing from the first token to
    the last token naturally retains the separators located between them. Leading,
    trailing, and separator-only regions are ignored because they contain no content.

    Args:
        text: Complete canonical document text.
        start: Inclusive region offset.
        end: Exclusive region offset.
        separator: Compiled expression matching text between structural units.

    Returns:
        Ordered, content-bearing, half-open character ranges.
    """
    ranges: list[tuple[int, int]] = []
    cursor = start

    # cursor marks the inclusive beginning of the next possible content region. Each
    # separator match closes the region immediately before the separator begins.
    for match in separator.finditer(text, start, end):
        # Retain only a non-empty, content-bearing region. strip() is used for the
        # decision but does not alter the offsets of a retained canonical-text slice.
        if match.start() > cursor and text[cursor : match.start()].strip():
            ranges.append((cursor, match.start()))

        # Skip the matched separator so it cannot become a standalone structural unit.
        cursor = match.end()

    # No later match closes the tail, so explicitly retain content between the final
    # separator and the exclusive end of the requested region.
    if cursor < end and text[cursor:end].strip():
        ranges.append((cursor, end))

    return ranges


def _token_range_for_characters(
    tokenized: TokenizedText,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Map a half-open character interval to intersecting global token indices.

    Token offsets and character ranges can have gaps where spaces or separators are
    not represented as tokens. Binary searches locate the first token ending after
    ``start`` and the first token starting at or after ``end``. Those two positions
    form the smallest half-open token range covering all intersecting tokens.

    Args:
        tokenized: Reusable start/end character arrays for all document tokens.
        start: Inclusive character boundary.
        end: Exclusive character boundary.

    Returns:
        ``(token_start, token_end)`` using inclusive/exclusive global indices.
    """
    # Tokens ending exactly at start do not intersect [start, end), while tokens
    # starting exactly at end also do not intersect it. The bisect variants encode
    # those half-open boundary rules. Both searches reuse arrays created once during
    # tokenization instead of rebuilding two document-sized lists for every text unit.
    token_start = bisect_right(tokenized.token_ends, start)
    token_end = bisect_left(tokenized.token_starts, end)
    return token_start, token_end
