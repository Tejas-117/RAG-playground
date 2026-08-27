# Chunking Implementation

The backend implements three document-local strategies over persisted
`document_parse.normalized_text`:

- `recursive` (default)
- `fixed_size`
- `paragraph`

Chunking does not parse source files again and never combines text from separate
documents. A persisted background worker invokes chunking through
`PipelineExecutor`, which builds or reuses the exact artifact before handing it
to embedding.

## Shared Flow

```text
POST /runs + resolved PipelineConfig
    -> persist pending pipeline_run and return 202
    -> worker claims run and marks chunking active
    -> pass corpus ID + resolved ChunkingConfig to the chunking service
    -> load ordered documents and canonical parses
    -> calculate compatibility fingerprint
    -> reuse matching ready chunk_set, if present
    -> tokenize and chunk every document independently
    -> attach page, block, parser, and document provenance
    -> atomically persist chunk_set and chunks
    -> link chunk_set from pipeline_run
    -> advance the run to embedding
```

The process is:

1. Validate the request and persist its immutable effective configuration with
   `pending` status, then mark the run `running`.
2. Load the corpus documents in their persisted order, together with each
   document's canonical parse text, pages, blocks, and parser metadata.
3. Reject an unknown corpus, an empty corpus, or any document without a
   completed canonical parse.
4. Resolve the configured strategy and the fixed backend tokenizer.
5. Fingerprint every input that can affect the output. Return the existing
   ready chunk set when that exact fingerprint has already been persisted.
6. Pass each document's `normalized_text` to the selected chunker separately.
   The chunker returns exact text spans with character and token offsets.
7. Build reusable page and block range indexes for the document. Convert each
   span into a persistable chunk, assign its deterministic ID and document-local
   ordinal, then derive provenance through indexed offset intersection.
8. Persist the chunk set and all of its chunks in one SQLite transaction, then
   read the completed artifact through the repository boundary.
9. Store the ready chunk-set ID, reuse flag, and chunking duration on the run,
   then advance its `current_stage` to `embedding`.

Chunk ordinals start at zero for every source document. Documents are never
combined, so chunk overlap cannot cross from one document into another.

Chunk-set and chunk IDs use deterministic UUID5 values. The fingerprint includes
the corpus and ordered document/parse IDs, resolved chunking configuration,
chunker name/version, tokenizer identifier/revision/digest/special-token policy,
and character safety limit. Identical ready artifacts are reused.

The builder reports whether the current request created or reused the artifact.
Concurrent requests calculating the same fingerprint converge on the single
ready database artifact even if both performed the in-memory chunking work.

Chunk-set and per-document lifecycle events include bounded input counts,
output counts, durations, reuse decisions, and stable identifiers. See
[logging_docs.md](logging_docs.md) for the logging configuration and privacy
rules.

## Pipeline Execution and Failures

`PipelineExecutor` owns stage ordering and the overall run lifecycle. It does
not implement tokenization or chunk boundaries; those remain in the chunking
service and strategy classes. It passes the ready artifact to the implemented
embedding/index service. Retrieval, generation, and evaluation remain future
stages.

Chunking runs in a local background worker because tokenization and SQLite work
are synchronous. `POST /runs` returns the persisted pending run immediately.
`GET /runs/{run_id}` exposes queued and active stage state, then the ready chunk
set ID, count, reuse flag, and measured duration. Chunk bodies remain internal
to the embedding stage.

Failures after run creation are persisted with `failed` status, a stable error
code, safe structured details, and timing. Chunk-set writes remain atomic, so a
failed run cannot reference a partial artifact.

## Shared Chunk Rules

Each normal chunk:

- is a non-empty exact slice of canonical text;
- stores zero-based, end-exclusive character offsets;
- stays within `chunk_size_tokens`;
- stays within 32,000 Unicode characters;
- stores global document token offsets; and
- retains intersecting page range and source-block ordinals.

The 32,000-character limit protects against a WordPiece `[UNK]` token spanning
an enormous URL, identifier, or generated string. Such input is split at Python
Unicode character boundaries. These fallback chunks retain exact character
offsets but store token offsets as `NULL` because one tokenizer token was split.

## Token-Boundary Representation and Performance

The tokenizer encodes each document once per chunk-set build. Its output is kept
as two parallel immutable tuples:

```text
token_starts[token_index] -> inclusive canonical character start
token_ends[token_index]   -> exclusive canonical character end
```

Both tuples contain one integer per tokenizer-visible token. A token index can
therefore retrieve both character boundaries without constructing a separate
long-lived object for every token. A compatibility `offsets` view remains
available for injected tokenizer adapters, but the production chunkers use the
parallel tuples directly.

Paragraph and recursive separators are discovered in character space and then
mapped into token space. The token-start and token-end tuples are constructed
once and reused by every mapping. Binary search finds the intersecting token
range for each paragraph, sentence, line, or whitespace unit. The implementation
does not rebuild document-sized boundary lists for each unit.

Tokenization is not persisted in this implementation. A different chunking
configuration or strategy therefore tokenizes the document again, while an
identical configuration can reuse its complete fingerprinted chunk set. A
persistent token-boundary artifact is deferred until measurements show that the
remaining tokenizer cost warrants the additional storage lifecycle.

## Fixed-Size Strategy

Fixed-size chunking encodes the document once and emits windows of
`chunk_size_tokens`. The stride is:

```text
chunk_size_tokens - chunk_overlap_tokens
```

Adjacent normal chunks therefore share the exact configured number of tokens.
The algorithm stops after the window containing the final token and does not
create a redundant overlap-only tail.

## Recursive Strategy

Recursive chunking discovers the smallest natural units required to fit the
limits using this fallback order:

```text
paragraph -> sentence -> line -> whitespace/word -> token
```

The discovered unit endings are sorted once. For each output window, binary
search selects the furthest ending within the token and character limits instead
of scanning all endings in the document. The next window begins at the prior
token end minus the configured overlap. Endings prefer natural boundaries; an
overlapped start may occur inside a sentence so the overlap remains exact.

Sentence recognition includes `.`, `!`, `?`, `。`, `！`, and `？`, with or
without following whitespace.

## Paragraph Strategy

Paragraph chunking treats blank lines as paragraph boundaries. A single newline
remains inside the paragraph. Adjacent complete paragraphs are greedily packed
while both hard limits hold, retaining their original internal separators.

An individually oversized paragraph is split into non-overlapping token windows.
The strategy always resolves overlap to `0` and rejects an explicit non-zero
overlap.

## Persisted Provenance

Each `chunk` row stores its exact text, document-relative ordinal, character and
optional token offsets, page range, and stable source document ID.
`source_metadata_json` contains:

- parse ID and parser name/version;
- original filename, MIME type, and content hash;
- parse-level metadata; and
- intersecting canonical block ordinals.

Pages, blocks, and chunks all refer to half-open character ranges within the
same canonical `normalized_text`: `[character_start_offset,
character_end_offset)`. A page or block intersects a chunk when the two ranges
share at least one character:

```text
chunk_start < source_end AND source_start < chunk_end
```

Page and block ranges are indexed once for each document. Their ordered end
offsets locate the first possible intersection through binary search, after
which only nearby ranges are examined until the chunk end is reached. If legacy
parse ranges are overlapping or out of order, provenance automatically uses the
original full-scan rule so persisted results remain correct.

Chunk boundaries are selected by the configured token or paragraph rules, not
by source-layout boundaries. Therefore one chunk can intersect multiple pages
or blocks. For example, a chunk covering `[450, 650)` intersects a page covering
`[0, 500)` and the next page covering `[502, 1000)`. The persisted `page_start`
and `page_end` record that page range, while `block_ordinals` records every
overlapping parsed block. Block text is not duplicated in metadata because the
chunk already stores its exact canonical text and the offsets link it back to
the parse artifact.

When a parser provides no page or block ranges, the corresponding provenance is
empty: page fields are `NULL` and `block_ordinals` is an empty list.

`section_path_json` remains `NULL` because section-aware parsing and chunking are
deferred. The complete artifact is written in one SQLite transaction, so a
child-write failure cannot leave a partial chunk set.
