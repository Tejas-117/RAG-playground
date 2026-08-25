# Parsing and Canonicalization

This document describes how an uploaded source file becomes a reusable parsed
artifact. Parsing happens once during upload. Later pipeline stages should read
the persisted canonical text instead of parsing the original file again.

## End-to-End Flow

```text
Uploaded file
    |
    v
Format-specific parser
    |
    v
ParsedDocument / ParsedPage / ParsedBlock
    |
    v
canonicalize_parsed_document()
    |
    v
CanonicalDocument / CanonicalPage / CanonicalBlock
    |
    v
_insert_parse_artifact()
    |
    v
SQLite: document_parse / parsed_page / parsed_block
```

The `POST /uploads` route performs this flow synchronously:

1. Validate the corpus name and every uploaded file.
2. Select a parser from the parser registry based on the file extension.
3. Verify the file size and confirm that its bytes match the declared format.
4. Save each file under `backend/uploads/` while calculating its SHA-256 hash.
5. Run the selected parser outside the FastAPI event loop.
6. Canonicalize the parser output into one authoritative text representation.
7. After every file has parsed successfully, persist the corpus, documents,
   canonical text, pages, and blocks in one SQLite transaction.

Documents in a multi-file upload are saved and parsed sequentially. The corpus
is not committed until every document has parsed successfully. If validation,
parsing, canonicalization, or persistence fails, all files saved for that
request are removed and no partial corpus is retained.

Upload and parsing lifecycle events, counts, durations, and safe failures are
written through the centralized backend logger. See
[logging_docs.md](logging_docs.md) for event names and privacy rules.

## Parser Output

Each format-specific parser returns a transient `ParsedDocument`. Depending on
the format, it may also contain `ParsedPage` and `ParsedBlock` instances.

Conceptually, parser output looks like this:

```python
ParsedDocument(
    text="Complete parser-provided text",
    parser_name="example-parser",
    parser_version="1.0.0",
    pages=[
        ParsedPage(
            page_number=1,
            text="Complete page text",
            blocks=[
                ParsedBlock(text="First block"),
                ParsedBlock(text="Second block"),
            ],
        )
    ],
)
```

At this stage, text can exist at the document, page, and block levels. These
objects represent parser output, not the final persisted storage layout. The
canonicalization step removes the need to persist duplicate page and block
text.

Page-aware parsers use the ordered pages and blocks to establish the canonical
reading order. Page-less formats, such as plain text, use
`ParsedDocument.text` directly.

## Building Canonical Page Text

`_build_page_text()` constructs the canonical text for one page and calculates
block offsets relative to the beginning of that page.

The function:

1. Ignores blocks whose text contains only whitespace.
2. Removes leading and trailing whitespace from each readable block while
   retaining meaningful newlines inside the block.
3. Joins readable blocks with the neutral separator `"\n\n"`.
4. Records the start and end offset of each block in the resulting page text.
5. Uses `ParsedPage.text` as a fallback when the page has no readable blocks.

For example:

```text
Block 1: "First paragraph"
Block 2: "Second paragraph"
```

becomes:

```text
First paragraph\n\nSecond paragraph
```

The block locations use half-open ranges, where the start is included and the
end is excluded:

```text
Block 1: [0, 15)
Block 2: [17, 33)
```

The separator occupies positions 15 and 16. `_build_page_text()` returns the
page text and block specifications containing these page-local offsets. It does
not create `CanonicalBlock` objects yet because their final offsets must be
relative to the complete document.

When the fallback page text is used, there are no block specifications because
the parser did not provide source-aware blocks whose locations could be
preserved.

## Building Canonical Document Text

`_canonicalize_pages()` processes all physical or logical pages in their parser
order. For each page, it:

1. Calls `_build_page_text()`.
2. Joins non-empty page contents with `"\n\n"`.
3. Tracks the page's start and end positions in the complete document.
4. Converts page-local block offsets into document-global offsets.
5. Assigns each block a document-wide ordinal.
6. Creates the corresponding `CanonicalPage` and `CanonicalBlock` instances.

For example, two page texts:

```text
Page 1: "Hello world"
Page 2: "Second page"
```

produce this normalized document text:

```text
Hello world\n\nSecond page
```

Their document-global ranges are:

```text
Page 1: [0, 11)
Page 2: [13, 24)
```

If a block on page 2 has page-local range `[0, 6)`, its document-global range
becomes `[13, 19)`. Therefore:

```python
normalized_text[13:19] == "Second"
```

Empty pages remain represented by `CanonicalPage` so their metadata and page
number are not lost. They do not add repeated separators to the normalized
text.

The result of `_canonicalize_pages()` is the complete normalized text and an
ordered list of canonical pages containing their canonical blocks.

## Validating the Canonical Document

`canonicalize_parsed_document()` is the public entry point for canonicalization.
It performs the following work:

- Requires a non-empty parser name and parser version for provenance.
- Uses `_canonicalize_pages()` when the parser supplied pages.
- Uses `ParsedDocument.text` directly when the parser supplied no pages.
- Rejects documents without extractable non-whitespace text.
- Rejects normalized UTF-8 text larger than 50 MiB.
- Verifies that document, page, and block metadata, warnings, and bounding boxes
  can be serialized as valid JSON.
- Returns an immutable `CanonicalDocument` containing normalized text, parser
  provenance, metadata, warnings, and canonical pages.

Conceptually, the result looks like this:

```python
CanonicalDocument(
    normalized_text="Complete canonical text",
    parser_name="example-parser",
    parser_version="1.0.0",
    pages=[
        CanonicalPage(
            page_number=1,
            character_start_offset=0,
            character_end_offset=100,
            blocks=[
                CanonicalBlock(
                    ordinal=0,
                    page_number=1,
                    character_start_offset=0,
                    character_end_offset=30,
                )
            ],
        )
    ],
)
```

`CanonicalPage` and `CanonicalBlock` deliberately contain offsets and
provenance instead of copied text. Their content is represented by slices of
`CanonicalDocument.normalized_text`.

## Purpose and Lifetime of the Canonical Objects

`CanonicalDocument`, `CanonicalPage`, and `CanonicalBlock` are temporary,
typed transfer objects between parsing and persistence. They provide one
validated structure for all parser implementations, regardless of the source
format.

The Python instances themselves are not stored. The upload route passes the
`CanonicalDocument` to the corpus repository, where `_insert_parse_artifact()`
converts the object tree into relational rows:

| Canonical object | SQLite table | Persisted information |
| --- | --- | --- |
| `CanonicalDocument` | `document_parse` | Full normalized text, parser identity, document metadata, warnings, counts, sizes, and duration |
| `CanonicalPage` | `parsed_page` | Page number, document-global character offsets, and page metadata |
| `CanonicalBlock` | `parsed_block` | Reading-order ordinal, source block index, document-global offsets, bounding box, and block metadata |

After the request finishes, the in-memory canonical objects can be discarded.
The SQLite records are the reusable parsed artifact.

## Reading Persisted Content

The complete parsed text is stored once in `document_parse.normalized_text`.
Page and block rows store offsets into that string, so their text does not need
to be duplicated in SQLite.

Page and block content can be reconstructed with normal string slicing:

```python
page_text = normalized_text[
    page["character_start_offset"] : page["character_end_offset"]
]

block_text = normalized_text[
    block["character_start_offset"] : block["character_end_offset"]
]
```

The internal `get_parsed_document()` repository function reads the normalized
text, pages, blocks, metadata, warnings, and parser identity from SQLite and
reconstructs an ordered dictionary representation. The backend does not
currently expose this complete artifact through an HTTP route.

## API Parse Summary

The upload response and `GET /corpora/` return a compact parse summary for each
document. The summary contains:

- Parse identifier
- Parser name and version
- Parser warnings
- Page and block counts
- UTF-8 text size and character count
- Parse duration in milliseconds
- Parse creation timestamp

Legacy documents without a parse artifact can have `parse: null`. Documents
uploaded through the current parsing flow always receive a parse artifact when
the upload succeeds.

## Why Section-Aware Parsing Is Deferred

The current parsing flow does not infer headings, semantic block roles, section
boundaries, or a section hierarchy. It preserves canonical text, pages, source
blocks, offsets, bounding boxes, and parser provenance instead.

Model-based PDF layout classification was evaluated as a way to identify titles
and section headings. It is comparatively slow for the synchronous upload flow,
particularly for long or visually complex PDFs. Running that analysis during
every upload would increase ingestion latency before chunking can begin.

A lighter PDF implementation would need to infer sections from embedded
bookmarks or heuristics such as font size, font weight, text position, and
repeated headers. PDFs do not expose one reliable heading structure, so these
rules would produce inconsistent results across documents. Libraries may also
use similar heuristics internally; using a library does not by itself guarantee
correct section detection.

Markdown headings and DOCX heading styles are easier to identify, but supporting
sections only for those formats would give chunking different structural
semantics depending on the source format. Section-aware parsing is therefore
deferred until its accuracy, latency, and cross-format behavior can be designed
and evaluated together.

This decision does not prevent paragraph chunking. Paragraph chunking operates
on paragraph boundaries in the persisted canonical text and does not require
semantic roles or a section hierarchy. Semantic chunking, which uses model-based
meaning or similarity to choose boundaries, is also outside the current scope
and is distinct from section-aware parsing.

## How Chunking Uses the Artifact

The chunk-set builder loads the persisted canonical artifact rather than parsing
the source file again. Each strategy splits `normalized_text` directly. The
builder intersects page and block ranges with each chunk range to attach:

- Source document identifier
- Page range
- Intersecting source-block ordinals
- Parser and source metadata

This layout gives chunkers one stable text representation while retaining the
source structure required for citations, metadata filtering, and source
highlighting. It also avoids storing duplicate copies of the page and block
text. Section paths remain `NULL` because section-aware parsing is deferred.

Streaming parsing is not part of the current implementation. Parsing currently
finishes in memory before the canonical artifact is written to SQLite.
