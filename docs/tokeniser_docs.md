# Chunking Tokeniser Decision

## Decision

The chunking stage will use the Hugging Face `tokenizers` runtime with the
`google-bert/bert-base-multilingual-cased` WordPiece tokeniser.

The tokeniser is a fixed backend implementation detail. It is not selected by
the user and does not change when the embedding provider or model changes.

Only the tokeniser definition is required. The BERT model configuration and
model weights must not be downloaded because the chunking stage does not run
the BERT model.

## Role in the Pipeline

The tokeniser is used only to measure chunk size and overlap and to translate
token boundaries back to offsets in the persisted canonical text:

```text
Persisted canonical text
    -> fixed chunking tokeniser
    -> token and character boundaries
    -> persisted chunk text and provenance
    -> selected embedding provider tokenises the chunk for its own model
    -> embedding vector
```

These are two separate tokenisation operations:

- The chunking tokeniser creates stable, reusable chunk boundaries.
- The embedding adapter later uses the selected model's own tokeniser to
  produce that model's input IDs.

Chunks remain ordinary slices of `document_parse.normalized_text`. Token IDs
from the chunking tokeniser are not embedded and do not replace the original
text.

## Why This Tokeniser Was Chosen

The selected tokeniser meets the requirements of the chunking artifact model:

- **Deterministic and offline:** the committed `tokenizer.json` contains the
  normaliser, pre-tokeniser, WordPiece vocabulary, and post-processing rules.
  Runtime chunking does not need network access.
- **Independent of embedding providers:** one corpus and chunking configuration
  produces the same chunk set for Ollama and future providers. Changing the
  embedding model can therefore reuse the chunk set and build a new vector
  index from it.
- **Exact source offsets:** Hugging Face `Encoding` results expose character
  offsets for each token. The chunker can use those offsets to persist exact
  half-open ranges into the canonical parse text.
- **Multilingual coverage:** multilingual BERT was trained over 104 languages
  and includes handling for scripts that do not use spaces between words. This
  is safer for user-provided documents than an English-oriented regex or
  whitespace counter.
- **Case preservation:** the cased variant does not lowercase input while
  calculating boundaries.
- **Lightweight integration:** the backend needs the `tokenizers` package and a
  roughly 1.9 MB JSON asset, not the full `transformers` package or BERT model.
- **Permissive licensing:** the upstream model repository identifies the asset
  as Apache 2.0 licensed.

References:

- [Multilingual BERT model card](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [Hugging Face Tokenizer API](https://huggingface.co/docs/tokenizers/main/api/tokenizer)
- [Hugging Face Encoding and offset API](https://huggingface.co/docs/tokenizers/main/api/encoding)

## Why Other Options Were Not Selected

### Model-specific tokenisers

A model-specific tokeniser would make chunk boundaries depend on the embedding
model. Changing from one embedding model to another could then require a new
chunk set, contrary to the planned artifact reuse boundary:

```text
chunk set -> embedding/index artifact A
          -> embedding/index artifact B
```

Embedding adapters will still use their exact model tokenisers and input limits
when vectors are created.

### Custom word or regular-expression counter

A custom counter would be small and easy to control, but its counts would not
represent subword tokens. It would also behave poorly for some multilingual
text, punctuation-heavy content, URLs, and scripts without whitespace-delimited
words.

### `tiktoken`

`tiktoken` is a mature BPE implementation, but its standard vocabularies are
primarily associated with OpenAI models. Selecting one as the universal
chunking vocabulary would add an unnecessary OpenAI bias to an Ollama-first,
provider-neutral pipeline.

### SentencePiece

SentencePiece is language independent and suitable for offline operation, but
the project would need to select or train, distribute, and version a separate
SentencePiece model. Its source-offset integration would also require more
custom handling than the offsets returned directly by Hugging Face
`tokenizers`.

## Known Tradeoffs

The fixed WordPiece count is a stable chunking measurement, not an exact count
for every embedding model. An embedding adapter may produce a different number
of tokens from the same chunk.

WordPiece can also emit one `[UNK]` token for an unusually long or unsupported
unbroken sequence. Chunkers must therefore retain a versioned character-length
safety limit and a character-boundary fallback for oversized units such as long
URLs or generated identifiers. This fallback must not split a Unicode code
point and must preserve exact canonical-text offsets.

## Asset Location

Save the tokeniser at this repository-relative path:

```text
backend/src/backend/chunking/assets/bert-base-multilingual-cased/tokenizer.json
```

This location makes the file a versioned backend implementation asset. It must
not be placed under `data/`, because `data/` is reserved for local fixtures and
must not contain runtime dependencies. It must also not be placed in the upload
directory, because it is application code data rather than user content.

The future chunking package should load the asset relative to its Python module,
not from the process working directory. The backend package configuration must
include the JSON file when a wheel is built.

## Pinned Download and Verification

Download the asset from the exact upstream commit that introduced the current
tokeniser file. Do not download from the mutable `main` branch.

```bash
mkdir -p \
  backend/src/backend/chunking/assets/bert-base-multilingual-cased

curl --fail --location \
  --output \
  backend/src/backend/chunking/assets/bert-base-multilingual-cased/tokenizer.json \
  https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/\
0fcb34d393e71211e8d72b52c31a46e7b7597068/tokenizer.json
```

Verify the downloaded file before committing it:

```bash
sha256sum \
  backend/src/backend/chunking/assets/bert-base-multilingual-cased/tokenizer.json
```

Expected result:

```text
f4a4d5bf7301717e261fafbe26e1eb967f6ba4cb3ae0ab7a29f4642ec229f386
```

The verified file size is `1,961,828` bytes. The pinned upstream source is:

```text
https://huggingface.co/google-bert/bert-base-multilingual-cased/blob/
0fcb34d393e71211e8d72b52c31a46e7b7597068/tokenizer.json
```

## Runtime Rules

The chunking implementation must follow these rules:

1. Load the committed local `tokenizer.json`; never call
   `Tokenizer.from_pretrained()` at runtime.
2. Encode with `add_special_tokens=False`. `[CLS]` and `[SEP]` are model input
   markers and must not consume the configured chunk budget.
3. Disable tokenizer truncation and padding. The chunker, not the tokeniser,
   owns windowing and overlap.
4. Use the encoding offsets to slice the original canonical text. Do not decode
   token IDs to construct persisted chunk text because that could alter
   whitespace or punctuation.
5. Persist zero-based, end-exclusive character offsets into
   `document_parse.normalized_text` and token offsets where the strategy can
   define them accurately.
6. Preserve natural boundaries for recursive and paragraph/section-aware
   chunking; the tokeniser provides size measurement and oversized-unit
   fallback boundaries rather than replacing structure-aware splitting.

The intended loading pattern is:

```python
from pathlib import Path

from tokenizers import Tokenizer

asset_path = (
    Path(__file__).parent
    / "assets"
    / "bert-base-multilingual-cased"
    / "tokenizer.json"
)
tokenizer = Tokenizer.from_file(str(asset_path))
encoding = tokenizer.encode(canonical_text, add_special_tokens=False)
```

The final implementation must wrap this library object behind a small typed
backend interface rather than exposing it directly to each chunking strategy.

## Versioning and Fingerprints

The tokeniser identity is part of chunk-set compatibility even though it is not
user configurable. The chunk-set fingerprint must include at least:

- the normalized `ChunkingConfig`
- the immutable corpus and parse inputs
- the chunker implementation name and version
- the tokeniser identifier `bert-base-multilingual-cased`
- the pinned upstream revision
- the tokeniser asset SHA-256 digest

Any change to the tokeniser asset, normalisation behavior, special-token policy,
or boundary algorithm must change the chunker version or fingerprint inputs.
Existing ready chunk sets remain historical artifacts and must not be silently
reinterpreted with the new behavior.

## Embedding Compatibility

An 800-token chunk according to this fixed WordPiece tokeniser is not guaranteed
to be 800 tokens for every embedding model. Before embedding is implemented,
each embedding-model catalog entry must declare its supported input limit. The
embedding stage must use the model's own tokeniser or provider-reported limit to
reject incompatible configurations instead of silently truncating chunks.

This validation belongs to the embedding/index stage. It does not make the
chunking tokeniser model-specific and does not change persisted chunk text.
