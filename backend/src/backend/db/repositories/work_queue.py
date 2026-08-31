"""Shared SQLite queue claim for preparation requests and legacy pipeline runs."""

from datetime import datetime, timezone
from typing import Literal, TypedDict

from backend.db.connection import connect


class ClaimedWorkItem(TypedDict):
    """Identify one atomically claimed durable background job.

    Attributes:
        kind: Executor category required by the claimed row.
        id: Stable identifier of the claimed persistence record.
    """

    kind: Literal["prepared_index", "pipeline_run"]
    id: str


class WorkQueueClaimError(RuntimeError):
    """Report a shared-queue row that could not be atomically claimed."""


def _utc_timestamp() -> str:
    """Create the UTC timestamp used when shared work is claimed.

    Args:
        None.

    Returns:
        ISO-8601 UTC timestamp ending in ``Z``.
    """
    # Use the same second-level format as every persisted job lifecycle.
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def claim_next_pending_work_item() -> ClaimedWorkItem | None:
    """Claim the oldest pending job across both supported durable job types.

    Args:
        None.

    Returns:
        Claimed job identity, or ``None`` when both queues are empty.

    Raises:
        WorkQueueClaimError: If the selected pending row cannot be claimed.
    """
    started_at = _utc_timestamp()

    # One immediate transaction serializes selection across both logical queues.
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id, kind
            FROM (
                SELECT id, 'prepared_index' AS kind, created_at
                FROM prepared_index WHERE status = 'pending'
                UNION ALL
                SELECT id, 'pipeline_run' AS kind, created_at
                FROM pipeline_run WHERE status = 'pending'
            )
            ORDER BY created_at, kind, id
            LIMIT 1
            """
        ).fetchone()

        # Do not write when neither durable queue contains pending work.
        if row is None:
            return None

        # Each job type starts in chunking but has an independent lifecycle table.
        if row["kind"] == "prepared_index":
            cursor = connection.execute(
                """
                UPDATE prepared_index
                SET status = 'running', current_stage = 'chunking', started_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (started_at, row["id"]),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE pipeline_run
                SET status = 'running', current_stage = 'chunking', started_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (started_at, row["id"]),
            )

        # A serialized compare-and-set must update exactly the selected row.
        if cursor.rowcount != 1:
            raise WorkQueueClaimError(
                f"Background work '{row['id']}' could not be claimed."
            )

    return {"kind": row["kind"], "id": row["id"]}
