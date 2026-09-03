# Evaluation Datasets

Evaluation datasets are immutable, corpus-scoped inputs for future benchmark
execution. Import is intentionally separate from retrieval, generation, and
evaluation so one stable set of examples can be reused across experiments.

## Import format

Import a dataset with `POST /datasets` using multipart form fields:

- `name`: required display name with 1–100 non-whitespace characters.
- `corpus_id`: the existing corpus against which document labels are resolved.
- `file`: a UTF-8 `.json` file no larger than 30 MB.

The JSON file contains a non-empty `examples` array:

```json
{
  "examples": [
    {
      "question": "How is authentication configured?",
      "reference_answer": "Authentication is configured through...",
      "relevant_documents": ["authentication.pdf", "deployment.md"]
    }
  ]
}
```

`question` is required and cannot be blank. `reference_answer` is optional.
`relevant_documents` is optional and defaults to an empty array. Duplicate
examples are deliberately preserved as separate examples with different stable
IDs and ordinals.

## Document-name resolution

Document names are trimmed and matched exactly and case-sensitively against
`document.original_filename` within the selected corpus. A unique match becomes
a relationship to the stable document ID.

An unknown filename or a filename shared by multiple corpus documents is not
guessed. The label is skipped, the question remains in the dataset, and a
warning is persisted and returned. Each warning includes the example ID and
ordinal, supplied document name, a machine-readable code, and a displayable
message. A question may therefore have no resolved relevant documents; later
evaluation must treat such a question as unlabelled for retrieval metrics.

## Persistence

The relational model uses:

- `evaluation_dataset` for the name, corpus, source filename, source SHA-256,
  import warnings, and creation timestamp.
- `evaluation_example` for stable ordered questions and optional reference
  answers.
- `evaluation_example_relevant_document` for normalized stable document links.

The uploaded JSON is canonicalized into these tables and is not retained as a
separate file. Dataset creation is transactional, so an invalid write cannot
leave a partial dataset. Datasets are immutable after import.

## Management API

- `POST /datasets` imports and returns full dataset detail with warnings.
- `GET /datasets` lists newest-first summaries. Use optional `corpus_id` and
  case-insensitive `search` query parameters.
- `GET /datasets/{dataset_id}` returns ordered examples, resolved documents, and
  persisted import warnings.
- `DELETE /datasets/{dataset_id}` deletes an unreferenced dataset and returns
  `204`. Once benchmark records reference datasets, protected deletion returns
  `409 dataset_in_use` rather than removing historical inputs.

Benchmark execution, evaluation metric calculation, dataset editing, manual UI
authoring, and frontend integration are not part of this backend slice.
