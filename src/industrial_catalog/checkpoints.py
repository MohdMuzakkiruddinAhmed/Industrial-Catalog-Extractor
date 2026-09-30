"""Small SQLite checkpoint store for resumable document processing."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Status = Literal["pending", "running", "succeeded", "failed", "review"]


@dataclass(frozen=True)
class Checkpoint:
    document_id: str
    source_path: str
    source_sha256: str
    status: Status
    attempt: int = 0
    detail: dict[str, Any] | None = None


class CheckpointStore:
    """Transaction-safe checkpoint registry keyed by content hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    detail_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def upsert(self, checkpoint: Checkpoint) -> None:
        values = asdict(checkpoint)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints (
                    document_id, source_path, source_sha256, status,
                    attempt, detail_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    source_sha256=excluded.source_sha256,
                    status=excluded.status,
                    attempt=excluded.attempt,
                    detail_json=excluded.detail_json,
                    updated_at=excluded.updated_at
                """,
                (
                    values["document_id"],
                    values["source_path"],
                    values["source_sha256"],
                    values["status"],
                    values["attempt"],
                    json.dumps(values["detail"], sort_keys=True) if values["detail"] else None,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM checkpoints GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}
