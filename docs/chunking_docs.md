# Chunking Implementation

The backend implements three document-local strategies over persisted
`document_parse.normalized_text`:

- `recursive` (default)
- `fixed_size`
- `paragraph`

Chunking does not parse source files again and never combines text from separate
documents. `POST /runs` does not invoke chunking yet; the reusable builder is an
internal backend service for the later execution layer.

## Shared Flow

```text
corpus ID + resolved ChunkingConfig
    -> load ordered documents and canonical parses
    -> calculate compatibility fingerprint
    -> reuse matching ready chunk_set, if present
    -> tokenize and chunk every document independently
    -> attach page, block, parser, and document provenance
    -> atomically persist chunk_set and chunks
```

The process is:

1. Load the corpus documents in their persisted order, together with each
   document's canonical parse text, pages, blocks, and parser metadata.
2. Reject an unknown corpus, an empty corpus, or any document without a
   completed canonical parse.
3. Resolve the configured strategy and the fixed backend tokenizer.
4. Fingerprint every input that can affect the output. Return the existing
   ready chunk set when that exact fingerprint has already been persisted.
5. Pass each document's `normalized_text` to the selected chunker separately.
   The chunker returns exact text spans with character and token offsets.
6. Convert each span into a persistable chunk. Assign its deterministic ID and
   document-local ordinal, then derive its page and block provenance by offset
   intersection.
7. Persist the chunk set and all of its chunks in one SQLite transaction, then
   read the completed artifact through the repository boundary.

Chunk ordinals start at zero for every source document. Documents are never
combined, so chunk overlap cannot cross from one document into another.

Chunk-set and chunk IDs use deterministic UUID5 values. The fingerprint includes
the corpus and ordered document/parse IDs, resolved chunking configuration,
chunker name/version, tokenizer identifier/revision/digest/special-token policy,
and character safety limit. Identical ready artifacts are reused.

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

It greedily chooses the furthest discovered unit ending within the current token
window. The next window begins at the prior token end minus the configured
overlap. Endings prefer natural boundaries; an overlapped start may occur inside
a sentence so the overlap remains exact.

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
