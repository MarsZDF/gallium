"""SQLite database helpers for experiment tracking."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from ._types import Experiment

# Schema version for migrations
SCHEMA_VERSION = 2

CREATE_EXPERIMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    seed INTEGER,
    path TEXT,
    model TEXT,
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,
    image_hash TEXT,
    params TEXT DEFAULT '{}',
    starred INTEGER DEFAULT 0,
    notes TEXT,
    created_at TEXT NOT NULL
);
"""

# Migration from v1 to v2: add starred and notes columns
MIGRATION_V1_TO_V2 = [
    "ALTER TABLE experiments ADD COLUMN starred INTEGER DEFAULT 0",
    "ALTER TABLE experiments ADD COLUMN notes TEXT",
]

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_experiments_prompt ON experiments(prompt);
CREATE INDEX IF NOT EXISTS idx_experiments_seed ON experiments(seed);
CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(model);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
"""


def get_connection(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Get a database connection, creating the database if needed.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Database connection with row factory set.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for better concurrency (readers don't block writers)
    conn.execute("PRAGMA journal_mode=WAL")
    # Set busy timeout to 5 seconds for concurrent access
    conn.execute("PRAGMA busy_timeout=5000")

    return conn


def _run_migration(
    cursor: sqlite3.Cursor,
    from_version: int,
    to_version: int,
    migrations: list[str],
) -> bool:
    """Run a set of migration statements.

    Args:
        cursor: Database cursor.
        from_version: Starting version.
        to_version: Target version.
        migrations: List of SQL statements to execute.

    Returns:
        bool: True if migration was applied, False if skipped.
    """
    import contextlib

    for sql in migrations:
        with contextlib.suppress(sqlite3.OperationalError):
            # Column/table may already exist (e.g., fresh install with new schema)
            cursor.execute(sql)

    return True


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema.

    Creates tables and indexes if they don't exist.
    Handles schema migrations for upgrading existing databases.

    Migration system:
    - Each schema version has explicit migration code
    - Migrations are idempotent (safe to run multiple times)
    - Version number is only updated after successful migration

    Args:
        conn: Database connection.
    """
    cursor = conn.cursor()

    # Create schema version table
    cursor.execute(CREATE_SCHEMA_VERSION_TABLE)

    # Check current schema version
    cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
    row = cursor.fetchone()
    current_version = row[0] if row else 0

    # Create experiments table (uses current schema)
    cursor.execute(CREATE_EXPERIMENTS_TABLE)

    # Create indexes
    cursor.executescript(CREATE_INDEXES)

    # Apply migrations incrementally
    # Each migration block handles upgrading from version N to N+1

    # Migration v1 -> v2: add starred and notes columns
    if current_version < 2:
        _run_migration(cursor, 1, 2, MIGRATION_V1_TO_V2)
        cursor.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (2,)
        )

    # Future migrations would follow this pattern:
    # if current_version < 3:
    #     _run_migration(cursor, 2, 3, MIGRATION_V2_TO_V3)
    #     cursor.execute(
    #         "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
    #         (3,)
    #     )

    conn.commit()


def insert_experiment(
    conn: sqlite3.Connection,
    prompt: str,
    seed: Optional[int] = None,
    path: Optional[str] = None,
    model: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    duration_ms: Optional[int] = None,
    image_hash: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    starred: bool = False,
    notes: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> int:
    """Insert a new experiment into the database.

    Args:
        conn: Database connection.
        prompt: The text prompt used for generation.
        seed: Random seed used.
        path: File path to the image.
        model: Model name.
        width: Image width.
        height: Image height.
        duration_ms: Generation time in milliseconds.
        image_hash: SHA-256 hash of the image.
        params: Additional parameters dict.
        starred: Whether experiment is starred/favorited.
        notes: Optional notes about the experiment.
        created_at: Timestamp (defaults to now).

    Returns:
        int: The ID of the inserted experiment.
    """
    if created_at is None:
        created_at = datetime.now()

    if params is None:
        params = {}

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO experiments (
            prompt, seed, path, model, width, height,
            duration_ms, image_hash, params, starred, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt,
            seed,
            path,
            model,
            width,
            height,
            duration_ms,
            image_hash,
            json.dumps(params),
            1 if starred else 0,
            notes,
            created_at.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid or 0


def insert_experiments_batch(
    conn: sqlite3.Connection,
    experiments: list[dict[str, Any]],
) -> list[int]:
    """Insert multiple experiments in a single transaction.

    All inserts succeed or none do (atomic transaction with rollback on error).

    Args:
        conn: Database connection.
        experiments: List of experiment dicts with keys matching log() params.

    Returns:
        list[int]: List of inserted experiment IDs.

    Raises:
        sqlite3.Error: If any insert fails (transaction is rolled back).
    """
    ids = []
    cursor = conn.cursor()

    try:
        for exp in experiments:
            created_at = exp.get("created_at") or datetime.now()
            params = exp.get("params") or {}
            starred = exp.get("starred", False)
            notes = exp.get("notes")

            cursor.execute(
                """
                INSERT INTO experiments (
                    prompt, seed, path, model, width, height,
                    duration_ms, image_hash, params, starred, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.get("prompt", ""),
                    exp.get("seed"),
                    exp.get("path"),
                    exp.get("model"),
                    exp.get("width"),
                    exp.get("height"),
                    exp.get("duration_ms"),
                    exp.get("image_hash"),
                    json.dumps(params),
                    1 if starred else 0,
                    notes,
                    created_at.isoformat(),
                ),
            )
            ids.append(cursor.lastrowid or 0)

        conn.commit()
        return ids
    except Exception:
        conn.rollback()
        raise


def row_to_experiment(row: sqlite3.Row) -> Experiment:
    """Convert a database row to an Experiment object.

    Args:
        row: SQLite row with experiment data.

    Returns:
        Experiment: The experiment object.
    """
    params_str = row["params"]
    params = json.loads(params_str) if params_str else {}

    created_at_str = row["created_at"]
    created_at = datetime.fromisoformat(created_at_str)

    # Handle nullable starred column (for migration compatibility)
    starred_val = row["starred"]
    starred = bool(starred_val) if starred_val is not None else False

    return Experiment(
        id=row["id"],
        prompt=row["prompt"],
        seed=row["seed"],
        path=row["path"],
        model=row["model"],
        width=row["width"],
        height=row["height"],
        duration_ms=row["duration_ms"],
        image_hash=row["image_hash"],
        params=params,
        starred=starred,
        notes=row["notes"],
        created_at=created_at,
    )


# Supported filter operations
SUPPORTED_FILTERS = {
    "prompt__contains",
    "prompt__startswith",
    "prompt__exact",
    "prompt",  # alias for exact
    "seed",
    "model",
    "width",
    "height",
    "starred",
    "notes__contains",
    "created_after",
    "created_before",
}


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards in a string.

    Args:
        value: String to escape.

    Returns:
        Escaped string safe for use in LIKE patterns.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_query(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a SQL query from filter parameters.

    Args:
        filters: Dict of filter conditions.

    Returns:
        tuple: (SQL WHERE clause, list of parameters)

    Raises:
        InvalidFilterError: If an unsupported filter is used.
    """
    from .exceptions import InvalidFilterError

    conditions: list[str] = []
    params: list[Any] = []

    for key, value in filters.items():
        if key not in SUPPORTED_FILTERS:
            raise InvalidFilterError(
                f"Unsupported filter: '{key}'. "
                f"Supported filters: {', '.join(sorted(SUPPORTED_FILTERS))}"
            )

        if key == "prompt__contains":
            conditions.append("prompt LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(value)}%")
        elif key == "prompt__startswith":
            conditions.append("prompt LIKE ? ESCAPE '\\'")
            params.append(f"{_escape_like(value)}%")
        elif key in ("prompt__exact", "prompt"):
            conditions.append("prompt = ?")
            params.append(value)
        elif key == "seed":
            conditions.append("seed = ?")
            params.append(value)
        elif key == "model":
            conditions.append("model = ?")
            params.append(value)
        elif key == "width":
            conditions.append("width = ?")
            params.append(value)
        elif key == "height":
            conditions.append("height = ?")
            params.append(value)
        elif key == "starred":
            conditions.append("starred = ?")
            params.append(1 if value else 0)
        elif key == "notes__contains":
            conditions.append("notes LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(value)}%")
        elif key == "created_after":
            if isinstance(value, datetime):
                conditions.append("created_at >= ?")
                params.append(value.isoformat())
            else:
                conditions.append("created_at >= ?")
                params.append(str(value))
        elif key == "created_before":
            if isinstance(value, datetime):
                conditions.append("created_at <= ?")
                params.append(value.isoformat())
            else:
                conditions.append("created_at <= ?")
                params.append(str(value))

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return where_clause, params


def query_experiments(
    conn: sqlite3.Connection,
    filters: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    order_desc: bool = True,
) -> list[Experiment]:
    """Query experiments with optional filters.

    Args:
        conn: Database connection.
        filters: Dict of filter conditions.
        limit: Maximum number of results.
        offset: Number of results to skip (for pagination).
        order_desc: If True, order by created_at DESC.

    Returns:
        list[Experiment]: List of matching experiments.
    """
    filters = filters or {}
    where_clause, params = build_query(filters)

    # Validate order direction (whitelist to prevent any future injection risk)
    order = "DESC" if order_desc else "ASC"
    if order not in ("ASC", "DESC"):
        raise ValueError(f"Invalid order direction: {order}")
    query = f"SELECT * FROM experiments WHERE {where_clause} ORDER BY created_at {order}"

    # Use parameterized queries for LIMIT/OFFSET to prevent SQL injection
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
        if offset > 0:
            query += " OFFSET ?"
            params.append(offset)

    cursor = conn.cursor()
    cursor.execute(query, params)

    return [row_to_experiment(row) for row in cursor.fetchall()]


def delete_experiment(conn: sqlite3.Connection, experiment_id: int) -> bool:
    """Delete an experiment by ID.

    Args:
        conn: Database connection.
        experiment_id: ID of the experiment to delete.

    Returns:
        bool: True if an experiment was deleted, False if not found.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
    conn.commit()
    return cursor.rowcount > 0


def update_experiment(
    conn: sqlite3.Connection,
    experiment_id: int,
    *,
    starred: Optional[bool] = None,
    notes: Optional[str] = None,
) -> bool:
    """Update an experiment's starred status or notes.

    Args:
        conn: Database connection.
        experiment_id: ID of the experiment to update.
        starred: New starred status (if provided).
        notes: New notes (if provided).

    Returns:
        bool: True if an experiment was updated, False if not found.
    """
    updates = []
    params: list[Any] = []

    if starred is not None:
        updates.append("starred = ?")
        params.append(1 if starred else 0)

    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)

    if not updates:
        return False

    params.append(experiment_id)
    query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"

    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    return cursor.rowcount > 0
