# Future Scope

This document records capabilities that are intentionally deferred from the current RAG
Playground implementation. It describes intended behavior and design constraints, not features
that are available today.

## Text-Derived Metadata Enrichment

Text-derived metadata enrichment generates structured metadata from a document's persisted
canonical text. It is different from both metadata embedded in the source file and metadata
entered directly by the user:

| Metadata source | Example | Ownership |
| --- | --- | --- |
| System metadata | Document ID, filename, hash, upload time | Application |
| Parser-extracted metadata | Embedded title, author, subject, creation date | Source file |
| User-provided metadata | Department, document type, custom tags | User |
| Enriched metadata | Detected language, entities, topics, inferred type | Enrichment stage |

The parsing and canonicalization flow is documented in
[parsing_docs.md](parsing_docs.md). Enrichment should consume its persisted output instead of
being embedded in individual file parsers.

### Intended Pipeline Position

Enrichment is an optional, independently rerunnable stage:

```text
Uploaded document
    -> parse and persist canonical text
    -> enrich document metadata
    -> chunk canonical text
    -> propagate selected metadata to chunks
    -> embed and index chunks
```

Enrichment may also be run after chunks already exist:

```text
Persisted canonical text
    -> run or rerun enrichment
    -> save a new metadata revision
    -> update affected chunk metadata
    -> synchronize vector-store metadata
```

This separation allows extraction logic or models to change without requiring another upload or
parse. Metadata-only changes should not require rechunking or re-embedding when metadata is not
part of the text supplied to the embedding model.

## Planned Extraction Methods

### 1. Regular Expressions and Rules

Regex and rule-based extraction should provide the first deterministic layer. It is suitable for
fields with recognizable syntax or a controlled vocabulary, including:

- Effective, publication, and revision dates.
- Document, invoice, contract, policy, and product identifiers.
- Version numbers.
- Email addresses and explicitly labelled authors.
- Departments, regions, and document types from controlled dictionaries.
- Labelled fields such as `Author:`, `Version:`, and `Effective Date:`.

Rules should be versioned and tested against fixed document fixtures. A matched value should retain
the rule identity and the supporting text span when practical. Rules should prefer precise matches
over broad guesses so their results remain useful for metadata filtering.

### 2. Local NLP Extraction

Local NLP should add fields that benefit from linguistic analysis while keeping document content
on the user's machine. Candidate operations include:

- Language detection.
- Named-entity recognition for people, organizations, locations, products, and dates.
- Keyword and keyphrase extraction.
- Topic and document-type classification against controlled taxonomies.
- Normalization of equivalent names and labels.

The selected NLP implementation should run offline, expose its model and version, have acceptable
CPU and memory requirements, and be evaluated on repository-owned fixtures. Extracted entities and
keywords should be normalized and deduplicated before becoming filterable metadata.

### 3. Local LLM Structured Extraction

A local LLM may handle semantic fields that rules and smaller NLP models cannot reliably obtain.
Possible fields include:

- A normalized title when no trustworthy title is available.
- A short document summary.
- Document type and department classification.
- Topics and domain-specific attributes.
- Effective dates or relationships expressed in inconsistent prose.

The model should return data through a strict, typed schema rather than free-form text. Responses
must be validated before persistence, and invalid or unsupported values must not silently enter the
filterable metadata set. The extractor should record the model identifier, prompt version,
generation parameters, and confidence or review status where supported.

Long documents should not be truncated without an explicit policy. A future implementation may use
representative sections, per-chunk extraction followed by document-level aggregation, or a staged
summary before structured extraction.

No particular local NLP or LLM model is selected yet. Selection should be based on offline quality,
latency, resource requirements, supported platforms, and license compatibility.

## Recommended Hybrid Strategy

The enrichment stage should use the least expensive reliable method for each field:

1. Run deterministic regex and dictionary rules.
2. Run local NLP for language, entities, keywords, and taxonomy classification.
3. Invoke the local LLM only for configured fields that remain unresolved or require semantic
   interpretation.
4. Resolve competing candidates using source priority, confidence thresholds, and field-specific
   policies.
5. Present sensitive or low-confidence suggestions for user confirmation instead of treating them
   as authoritative.

User-provided values should not be overwritten by enrichment. Parser-extracted values should remain
available even when an enrichment method proposes a normalized or alternative value.

## Metadata Shape and Provenance

Metadata sources should remain logically separated:

```json
{
  "system": {
    "document_id": "doc_123",
    "filename": "remote-work-policy.pdf"
  },
  "parser": {
    "title": "Remote Work Policy",
    "authors": ["Jane Doe"]
  },
  "user": {
    "department": "human_resources"
  },
  "enrichment": {
    "language": "en",
    "document_type": "policy",
    "topics": ["remote_work", "information_security"]
  }
}
```

Rich enrichment records in SQLite should preserve how each value was produced:

```json
{
  "field": "document_type",
  "value": "policy",
  "method": "local_nlp_classifier",
  "extractor_version": "document-classifier-v1",
  "confidence": 0.91,
  "evidence": "Remote Work Policy",
  "created_at": "2026-08-11T10:00:00Z"
}
```

Depending on the method and field, provenance may include:

- Extraction method and implementation version.
- Rule, NLP model, or local LLM identifier.
- Prompt and schema version for LLM extraction.
- Confidence score or confirmation status.
- Supporting text or canonical character offsets when practical.
- Extraction timestamp and metadata revision.
- Structured warnings and failure details.

The vector store should receive only a validated, flattened projection of values needed for
filtering and source display. Full evidence, prompts, warnings, and extraction history should remain
in SQLite. The artifact and provenance rules in
[pipeline_architecture.md](pipeline/pipeline_architecture.md) continue to apply.

## Document-Level and Chunk-Level Enrichment

Document-level metadata describes the complete source, for example language, document type,
department, or publication date. Every indexed chunk may inherit the approved filterable subset of
these values.

Chunk-level metadata describes a particular part of the source, for example section heading,
entities mentioned in that section, or local topics. It should retain stable document and chunk IDs,
page or section provenance, and canonical offsets when available.

Document-level values should not be inferred by blindly copying one chunk's result. Aggregation
must define how repeated, conflicting, or low-confidence chunk predictions become document-level
metadata.

## Reruns, Backfills, and Index Synchronization

An enrichment configuration should be versioned so existing results can be reproduced and stale
metadata can be identified. Changing a rule set, model, taxonomy, prompt, or extraction schema
creates a new enrichment revision rather than silently rewriting provenance.

A future backfill operation should:

1. Read the persisted canonical text for selected documents.
2. Run the configured extractors without reparsing the source files.
3. Validate and persist the new metadata revision.
4. Propagate approved document metadata to existing chunks.
5. Update the corresponding metadata in Chroma or another vector-store adapter.
6. Record per-document success, warnings, duration, and structured failures.

Metadata synchronization should be recoverable and idempotent. A failed enrichment or index update
must not corrupt the previous usable metadata revision.

## Filtering and User Experience

Only normalized, validated values should be exposed to metadata filtering. Controlled taxonomies
are preferred for fields such as department and document type so semantically equivalent labels do
not fragment filters.

The UI may eventually show:

- Extracted values grouped by system, parser, user, and enrichment source.
- Confidence, evidence, and extraction method for generated values.
- Suggestions awaiting confirmation.
- Warnings and failed enrichment fields.
- Controls to rerun enrichment with a newer configuration.

Generated metadata must be presented as a model or rule output, not as guaranteed fact. It should
not be used as the sole basis for authorization, access control, compliance, or destructive actions.
Metadata filters remain a later pipeline capability as described in
[pipeline_parameters.md](pipeline/pipeline_parameters.md).

## Future Implementation Phases

1. Define typed document- and chunk-enrichment contracts, metadata namespaces, provenance, and
   revision rules.
2. Implement and evaluate deterministic regex and dictionary extractors.
3. Add pluggable local NLP adapters for language, entities, keywords, and classification.
4. Add a schema-constrained local LLM adapter with resource limits and reproducible configuration.
5. Implement persisted enrichment artifacts, reruns, and backfills over canonical text.
6. Propagate approved metadata to chunks and synchronize vector-store metadata.
7. Expose metadata inspection, confirmation, filtering, and evaluation in the API and UI.

Every implementation phase should use deterministic offline tests, fixed document fixtures, mocked
model adapters where appropriate, and explicit cases for missing values, conflicting candidates,
invalid output, low confidence, empty documents, and extractor failures.
