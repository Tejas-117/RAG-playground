"""Pure retrieval scoring over persisted chunk document identities."""

from typing import Any

# These are the retrieval metrics implemented by the deterministic scorer.
METRICS = ("hit_rate_at_k", "recall_at_k", "mrr")


def score_question(
    relevant_document_ids: set[str], ranked_document_ids: list[str], top_k: int
) -> dict[str, Any]:
    """Score one question; return metrics and matching rank evidence.
    For now all the metrics are calculated and returned

    Args:
        relevant_document_ids: Distinct resolved source document labels.
        ranked_document_ids: Source document IDs in saved chunk rank order.
        top_k: Saved retrieval cutoff.

    Returns:
        Metric values and matching chunk ranks, or a skip reason without labels.
    """
    # Unlabelled examples do not contribute zeros to an aggregate.
    if not relevant_document_ids:
        return {
            "skip_reason": "no_resolved_document_labels",
            "scores": None,
            "matching_ranks": [],
            "relevant_document_ids": [],
        }

    # Repeated chunks retain their rank but count their document only once for recall.
    # Visit every saved chunk rank because duplicate documents remain meaningful for MRR.
    matching_ranks = [
        rank
        for rank, document_id in enumerate(ranked_document_ids[:top_k], 1)
        if document_id in relevant_document_ids
    ]
    matched_documents = set(ranked_document_ids[:top_k]) & relevant_document_ids
    return {
        "skip_reason": None,
        "scores": {
            "hit_rate_at_k": float(bool(matching_ranks)),
            "recall_at_k": len(matched_documents) / len(relevant_document_ids),
            "mrr": 1 / matching_ranks[0] if matching_ranks else 0.0,
        },
        "matching_ranks": matching_ranks,
        "relevant_document_ids": sorted(relevant_document_ids),
    }
