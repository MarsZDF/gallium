"""Experiment tracking with SQLite backend."""

import csv
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, Union

from ._db import (
    delete_experiment,
    get_connection,
    init_db,
    insert_experiment,
    insert_experiments_batch,
    query_experiments,
    update_experiment,
)
from ._imaging import compute_image_hash
from ._types import Experiment
from .exceptions import SerializationError

# Default database path
DEFAULT_DB_PATH = "./gallium.db"


class Tracker:
    """Experiment tracker with SQLite backend.

    Manages a database of image generation experiments for tracking,
    querying, and exporting experiment history.

    Args:
        db_path: Path to the SQLite database file. Defaults to "./gallium.db".

    Example:
        >>> tracker = Tracker("my_experiments.db")
        >>> tracker.log(prompt="cat", seed=42, path="cat.png")
        >>> experiments = tracker.find(prompt__contains="cat")
        >>> tracker.export("json", path="experiments.json")
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        """Initialize the tracker.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure database is initialized (lazy initialization)."""
        if not self._initialized:
            self._conn = get_connection(self.db_path)
            init_db(self._conn)
            self._initialized = True

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection, initializing if needed."""
        self._ensure_initialized()
        assert self._conn is not None  # Ensured by _ensure_initialized
        return self._conn

    def log(
        self,
        prompt: str,
        *,
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
    ) -> int:
        """Log an image generation experiment.

        Args:
            prompt: The text prompt used for generation.
            seed: Random seed used for generation.
            path: File path to the generated image.
            model: Name of the model used (e.g., "flux.2-pro").
            width: Image width in pixels.
            height: Image height in pixels.
            duration_ms: Generation time in milliseconds.
            image_hash: SHA-256 hash of the image. If omitted and path is provided,
                       the hash is computed automatically.
            params: Additional parameters as a JSON-serializable dict.
            starred: Whether to mark this as a favorite experiment.
            notes: Optional notes about the experiment.

        Returns:
            int: The ID of the logged experiment.

        Raises:
            SerializationError: If params contains non-JSON-serializable data.

        Example:
            >>> exp_id = tracker.log(
            ...     prompt="cyberpunk city",
            ...     seed=42,
            ...     path="output.png",
            ...     model="flux.2-pro",
            ...     params={"guidance": 7.5}
            ... )
        """
        # Validate params is JSON-serializable
        if params is not None:
            try:
                json.dumps(params)
            except (TypeError, ValueError) as e:
                raise SerializationError(f"params must be JSON-serializable: {e}") from e

        # Auto-compute image hash if path provided and hash not specified
        if path is not None and image_hash is None:
            image_hash = compute_image_hash(path)

        return insert_experiment(
            self._get_conn(),
            prompt=prompt,
            seed=seed,
            path=path,
            model=model,
            width=width,
            height=height,
            duration_ms=duration_ms,
            image_hash=image_hash,
            params=params,
            starred=starred,
            notes=notes,
        )

    def log_many(self, experiments: list[dict[str, Any]]) -> list[int]:
        """Log multiple experiments in a single transaction.

        Args:
            experiments: List of experiment dicts with keys matching log() params.

        Returns:
            list[int]: List of inserted experiment IDs.

        Raises:
            SerializationError: If any params contains non-JSON-serializable data.

        Example:
            >>> ids = tracker.log_many([
            ...     {"prompt": "cat", "seed": 1, "path": "cat1.png"},
            ...     {"prompt": "cat", "seed": 2, "path": "cat2.png"},
            ... ])
        """
        # Validate all params are JSON-serializable
        for exp in experiments:
            params = exp.get("params")
            if params is not None:
                try:
                    json.dumps(params)
                except (TypeError, ValueError) as e:
                    raise SerializationError(f"params must be JSON-serializable: {e}") from e

        # Auto-compute image hashes where needed
        processed = []
        for exp in experiments:
            exp_copy = dict(exp)
            path = exp_copy.get("path")
            if path is not None and exp_copy.get("image_hash") is None:
                exp_copy["image_hash"] = compute_image_hash(path)
            processed.append(exp_copy)

        return insert_experiments_batch(self._get_conn(), processed)

    def find(
        self,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        **filters: Any,
    ) -> list[Experiment]:
        """Find experiments matching filter criteria.

        Supported filters:
            - prompt__contains: Prompt contains substring
            - prompt__startswith: Prompt starts with string
            - prompt__exact or prompt: Exact prompt match
            - seed: Exact seed match
            - model: Exact model match
            - width: Exact width match
            - height: Exact height match
            - created_after: Created after datetime
            - created_before: Created before datetime

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            **filters: Filter conditions.

        Returns:
            list[Experiment]: Matching experiments, most recent first.

        Raises:
            InvalidFilterError: If an unsupported filter is used.

        Example:
            >>> experiments = tracker.find(prompt__contains="cat", seed=42)
            >>> # Pagination
            >>> page1 = tracker.find(limit=10, offset=0)
            >>> page2 = tracker.find(limit=10, offset=10)
        """
        # If DB doesn't exist yet, return empty list
        if not self._initialized and not self.db_path.exists():
            return []

        return query_experiments(self._get_conn(), filters, limit=limit, offset=offset)

    def recent(self, limit: int = 10) -> list[Experiment]:
        """Get the most recent experiments.

        Args:
            limit: Maximum number of experiments to return.

        Returns:
            list[Experiment]: Most recent experiments.

        Example:
            >>> recent = tracker.recent(5)
            >>> for exp in recent:
            ...     print(f"{exp.prompt} (seed={exp.seed})")
        """
        # If DB doesn't exist yet, return empty list
        if not self._initialized and not self.db_path.exists():
            return []

        return query_experiments(self._get_conn(), limit=limit)

    def export(
        self,
        format: str = "csv",
        path: Optional[str] = None,
        *,
        title: str = "Gallium Experiments",
        thumbnail_size: int = 256,
    ) -> str:
        """Export experiment data to a file.

        Args:
            format: Export format: "csv", "json", or "html".
            path: Output file path. Defaults to "gallium_export.{format}".
            title: Title for HTML export (ignored for csv/json).
            thumbnail_size: Max thumbnail dimension for HTML export.

        Returns:
            str: Path to the exported file.

        Raises:
            ValueError: If format is not "csv", "json", or "html".

        Example:
            >>> tracker.export("csv")  # Creates gallium_export.csv
            >>> tracker.export("json", path="my_experiments.json")
            >>> tracker.export("html", title="My Generations")
        """
        if format not in ("csv", "json", "html"):
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'html'.")

        if path is None:
            path = f"gallium_export.{format}"

        experiments = self.find()

        if format == "csv":
            self._export_csv(experiments, path)
        elif format == "json":
            self._export_json(experiments, path)
        else:
            self._export_html(experiments, path, title, thumbnail_size)

        return path

    def _export_csv(self, experiments: list[Experiment], path: str) -> None:
        """Export experiments to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "prompt", "seed", "path", "model", "width", "height",
                "duration_ms", "image_hash", "params", "starred", "notes", "created_at"
            ])
            for exp in experiments:
                writer.writerow([
                    exp.id,
                    exp.prompt,
                    exp.seed,
                    exp.path,
                    exp.model,
                    exp.width,
                    exp.height,
                    exp.duration_ms,
                    exp.image_hash,
                    json.dumps(exp.params),
                    exp.starred,
                    exp.notes,
                    exp.created_at.isoformat(),
                ])

    def _export_json(self, experiments: list[Experiment], path: str) -> None:
        """Export experiments to JSON."""
        data = []
        for exp in experiments:
            data.append({
                "id": exp.id,
                "prompt": exp.prompt,
                "seed": exp.seed,
                "path": exp.path,
                "model": exp.model,
                "width": exp.width,
                "height": exp.height,
                "duration_ms": exp.duration_ms,
                "image_hash": exp.image_hash,
                "params": exp.params,
                "starred": exp.starred,
                "notes": exp.notes,
                "created_at": exp.created_at.isoformat(),
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _export_html(
        self,
        experiments: list[Experiment],
        path: str,
        title: str,
        thumbnail_size: int,
    ) -> None:
        """Export experiments to HTML gallery."""
        import html as html_module
        from pathlib import Path as PathLib

        # HTML template with embedded CSS
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_module.escape(title)}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 2rem;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
            color: #fff;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 2rem;
            color: #888;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax({thumbnail_size + 40}px, 1fr));
            gap: 1.5rem;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .card {{
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        }}
        .card.starred {{
            border: 2px solid #ffd700;
        }}
        .card img {{
            width: 100%;
            height: {thumbnail_size}px;
            object-fit: cover;
            background: #0f0f23;
        }}
        .card .info {{
            padding: 1rem;
        }}
        .card .prompt {{
            font-size: 0.85rem;
            color: #ccc;
            margin-bottom: 0.5rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .card .meta {{
            font-size: 0.75rem;
            color: #666;
        }}
        .card .meta span {{
            margin-right: 0.75rem;
        }}
        .card .star {{
            color: #ffd700;
            margin-right: 0.25rem;
        }}
        .card .notes {{
            font-size: 0.75rem;
            color: #888;
            font-style: italic;
            margin-top: 0.5rem;
            border-top: 1px solid #333;
            padding-top: 0.5rem;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }}
        .modal.active {{ display: flex; }}
        .modal img {{
            max-width: 90vw;
            max-height: 90vh;
            object-fit: contain;
        }}
        .modal-close {{
            position: absolute;
            top: 1rem; right: 1rem;
            background: none;
            border: none;
            color: #fff;
            font-size: 2rem;
            cursor: pointer;
        }}
        .no-image {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: {thumbnail_size}px;
            background: #0f0f23;
            color: #444;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <h1>{html_module.escape(title)}</h1>
    <p class="stats">{len(experiments)} experiments</p>
    <div class="grid">
"""

        for exp in experiments:
            starred_class = " starred" if exp.starred else ""
            starred_icon = '<span class="star">★</span>' if exp.starred else ""

            # Build image tag
            if exp.path and PathLib(exp.path).exists():
                escaped_path = html_module.escape(exp.path)
                img_tag = f'<img src="{escaped_path}" alt="Experiment {exp.id}" loading="lazy">'
            else:
                img_tag = '<div class="no-image">No image</div>'

            # Build meta info
            meta_parts = []
            if exp.seed is not None:
                meta_parts.append(f"seed={exp.seed}")
            if exp.model:
                meta_parts.append(exp.model)
            meta_str = " · ".join(meta_parts) if meta_parts else ""

            # Notes section
            notes_html = ""
            if exp.notes:
                escaped_notes = html_module.escape(exp.notes)
                notes_html = f'<div class="notes">{escaped_notes}</div>'

            # Use JSON encoding for onclick to prevent XSS via malicious filenames
            import json as json_module
            onclick_path = json_module.dumps(exp.path or "")
            onclick_attr = f"openModal({onclick_path})"
            escaped_prompt = html_module.escape(exp.prompt)
            escaped_meta = html_module.escape(meta_str)
            html_content += f"""        <div class="card{starred_class}" onclick="{onclick_attr}">
            {img_tag}
            <div class="info">
                <div class="prompt">{starred_icon}{escaped_prompt}</div>
                <div class="meta"><span>#{exp.id}</span>{escaped_meta}</div>
                {notes_html}
            </div>
        </div>
"""

        html_content += """    </div>
    <div class="modal" id="modal" onclick="closeModal()">
        <button class="modal-close" onclick="closeModal()">&times;</button>
        <img id="modal-img" src="" alt="Full size">
    </div>
    <script>
        function openModal(src) {
            if (!src) return;
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').classList.add('active');
        }
        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
    </script>
</body>
</html>
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def delete(self, experiment_id: int) -> bool:
        """Delete an experiment by ID.

        Args:
            experiment_id: ID of the experiment to delete.

        Returns:
            bool: True if the experiment was deleted, False if not found.

        Example:
            >>> tracker.delete(42)
            True
        """
        if not self._initialized and not self.db_path.exists():
            return False

        return delete_experiment(self._get_conn(), experiment_id)

    def star(self, experiment_id: int, starred: bool = True) -> bool:
        """Star or unstar an experiment.

        Args:
            experiment_id: ID of the experiment to update.
            starred: True to star, False to unstar.

        Returns:
            bool: True if the experiment was updated, False if not found.

        Example:
            >>> tracker.star(42)  # Star experiment 42
            >>> tracker.star(42, False)  # Unstar experiment 42
        """
        if not self._initialized and not self.db_path.exists():
            return False

        return update_experiment(self._get_conn(), experiment_id, starred=starred)

    def annotate(self, experiment_id: int, notes: str) -> bool:
        """Add or update notes on an experiment.

        Args:
            experiment_id: ID of the experiment to update.
            notes: Notes to add (replaces existing notes).

        Returns:
            bool: True if the experiment was updated, False if not found.

        Example:
            >>> tracker.annotate(42, "Best result so far!")
            True
        """
        if not self._initialized and not self.db_path.exists():
            return False

        return update_experiment(self._get_conn(), experiment_id, notes=notes)

    def to_dataframe(self, **filters: Any) -> Any:
        """Export experiments to a pandas DataFrame.

        Requires pandas to be installed. Useful for data analysis and
        integration with Jupyter notebooks.

        Args:
            **filters: Optional filter conditions (same as find()).

        Returns:
            pandas.DataFrame: DataFrame with experiment data.

        Raises:
            MissingDependencyError: If pandas is not installed.

        Example:
            >>> df = tracker.to_dataframe()
            >>> df.head()
            >>> # With filters
            >>> df = tracker.to_dataframe(model="flux.2-pro")
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as e:
            from .exceptions import MissingDependencyError
            raise MissingDependencyError(
                "pandas is required for to_dataframe(). Install it with: pip install pandas"
            ) from e

        experiments = self.find(**filters)

        if not experiments:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=[
                "id", "prompt", "seed", "path", "model", "width", "height",
                "duration_ms", "image_hash", "params", "starred", "notes", "created_at"
            ])

        data = []
        for exp in experiments:
            data.append({
                "id": exp.id,
                "prompt": exp.prompt,
                "seed": exp.seed,
                "path": exp.path,
                "model": exp.model,
                "width": exp.width,
                "height": exp.height,
                "duration_ms": exp.duration_ms,
                "image_hash": exp.image_hash,
                "params": exp.params,
                "starred": exp.starred,
                "notes": exp.notes,
                "created_at": exp.created_at,
            })

        return pd.DataFrame(data)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    def __enter__(self) -> "Tracker":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context manager, closing the connection."""
        self.close()


# Global default tracker (lazy initialized, thread-safe)
_default_tracker: Optional[Tracker] = None
_default_tracker_lock = threading.Lock()


def _get_default_tracker() -> Tracker:
    """Get or create the default tracker (thread-safe)."""
    global _default_tracker

    # Fast path: already initialized
    if _default_tracker is not None:
        return _default_tracker

    # Slow path: need to initialize (with lock for thread safety)
    with _default_tracker_lock:
        # Double-check after acquiring lock
        if _default_tracker is None:
            _default_tracker = Tracker()
        return _default_tracker


def init(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
    """Initialize gallium with a custom database path.

    Call this before log() if you want to use a non-default database location.

    Args:
        db_path: Path to the SQLite database file.

    Example:
        >>> gallium.init("custom/path.db")
        >>> gallium.log(prompt="test", seed=1)
    """
    global _default_tracker
    _default_tracker = Tracker(db_path)


def log(
    prompt: str,
    *,
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
) -> int:
    """Log an image generation experiment.

    Auto-initializes the database on first call.

    Args:
        prompt: The text prompt used for generation.
        seed: Random seed used for generation.
        path: File path to the generated image.
        model: Name of the model used (e.g., "flux.2-pro").
        width: Image width in pixels.
        height: Image height in pixels.
        duration_ms: Generation time in milliseconds.
        image_hash: SHA-256 hash of the image. If omitted and path is provided,
                   the hash is computed automatically.
        params: Additional parameters as a JSON-serializable dict.
        starred: Whether to mark this experiment as a favorite.
        notes: Optional notes about the experiment.

    Returns:
        int: The ID of the logged experiment.

    Example:
        >>> gallium.log(
        ...     prompt="cyberpunk city",
        ...     seed=42,
        ...     path="output.png",
        ...     model="flux.2-pro"
        ... )
    """
    return _get_default_tracker().log(
        prompt,
        seed=seed,
        path=path,
        model=model,
        width=width,
        height=height,
        duration_ms=duration_ms,
        image_hash=image_hash,
        params=params,
        starred=starred,
        notes=notes,
    )


def log_many(experiments: list[dict[str, Any]]) -> list[int]:
    """Log multiple experiments in a single transaction.

    Args:
        experiments: List of experiment dicts with keys matching log() params.

    Returns:
        list[int]: List of inserted experiment IDs.

    Example:
        >>> gallium.log_many([
        ...     {"prompt": "cat", "seed": 1, "path": "cat1.png"},
        ...     {"prompt": "cat", "seed": 2, "path": "cat2.png"},
        ... ])
    """
    return _get_default_tracker().log_many(experiments)


def find(
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    **filters: Any,
) -> list[Experiment]:
    """Find experiments matching filter criteria.

    Supported filters:
        - prompt__contains: Prompt contains substring
        - prompt__startswith: Prompt starts with string
        - prompt__exact or prompt: Exact prompt match
        - seed: Exact seed match
        - model: Exact model match
        - width: Exact width match
        - height: Exact height match
        - created_after: Created after datetime
        - created_before: Created before datetime

    Args:
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
        **filters: Filter conditions.

    Returns:
        list[Experiment]: Matching experiments, most recent first.

    Example:
        >>> experiments = gallium.find(prompt__contains="cat")
        >>> # Pagination
        >>> page1 = gallium.find(limit=10, offset=0)
        >>> page2 = gallium.find(limit=10, offset=10)
    """
    return _get_default_tracker().find(limit=limit, offset=offset, **filters)


def recent(limit: int = 10) -> list[Experiment]:
    """Get the most recent experiments.

    Args:
        limit: Maximum number of experiments to return.

    Returns:
        list[Experiment]: Most recent experiments.

    Example:
        >>> recent = gallium.recent(5)
    """
    return _get_default_tracker().recent(limit)


def export(
    format: str = "csv",
    path: Optional[str] = None,
    *,
    title: str = "Gallium Experiments",
    thumbnail_size: int = 256,
) -> str:
    """Export experiment data to a file.

    Args:
        format: Export format: "csv", "json", or "html".
        path: Output file path. Defaults to "gallium_export.{format}".
        title: Title for HTML export (ignored for csv/json).
        thumbnail_size: Max thumbnail dimension for HTML export.

    Returns:
        str: Path to the exported file.

    Example:
        >>> gallium.export("csv")
        >>> gallium.export("json", path="experiments.json")
        >>> gallium.export("html", title="My Hackathon Results")
    """
    return _get_default_tracker().export(
        format, path, title=title, thumbnail_size=thumbnail_size
    )


def delete(experiment_id: int) -> bool:
    """Delete an experiment by ID.

    Args:
        experiment_id: ID of the experiment to delete.

    Returns:
        bool: True if the experiment was deleted, False if not found.

    Example:
        >>> gallium.delete(42)
        True
    """
    return _get_default_tracker().delete(experiment_id)


def star(experiment_id: int, starred: bool = True) -> bool:
    """Star or unstar an experiment.

    Args:
        experiment_id: ID of the experiment to update.
        starred: True to star, False to unstar.

    Returns:
        bool: True if the experiment was updated, False if not found.

    Example:
        >>> gallium.star(42)  # Star experiment 42
        >>> gallium.star(42, False)  # Unstar it
    """
    return _get_default_tracker().star(experiment_id, starred)


def annotate(experiment_id: int, notes: str) -> bool:
    """Add or update notes on an experiment.

    Args:
        experiment_id: ID of the experiment to update.
        notes: Notes to add (replaces existing notes).

    Returns:
        bool: True if the experiment was updated, False if not found.

    Example:
        >>> gallium.annotate(42, "Best result so far!")
        True
    """
    return _get_default_tracker().annotate(experiment_id, notes)


def sweep(
    param_name: str,
    values: Any,
    base_params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Generate parameter combinations for a sweep.

    Creates a list of parameter dicts varying one parameter across values.
    Useful for seed sweeps, guidance scale experiments, etc.

    Args:
        param_name: Name of the parameter to vary (e.g., "seed", "guidance").
        values: Iterable of values to sweep (e.g., range(1, 10), [5.0, 7.5, 10.0]).
        base_params: Base parameters to merge with (optional).

    Returns:
        list[dict]: List of parameter dicts ready for generation.

    Example:
        >>> # Seed sweep
        >>> params = gallium.sweep("seed", range(1, 10))
        >>> for p in params:
        ...     image = generate(**p)
        ...     gallium.log(**p, path=f"out_{p['seed']}.png")

        >>> # Guidance scale sweep with base params
        >>> base = {"prompt": "cyberpunk city", "model": "flux.2-pro"}
        >>> params = gallium.sweep("guidance", [5.0, 7.5, 10.0], base)
        >>> for p in params:
        ...     print(p)
        {'prompt': 'cyberpunk city', 'model': 'flux.2-pro', 'guidance': 5.0}
        {'prompt': 'cyberpunk city', 'model': 'flux.2-pro', 'guidance': 7.5}
        {'prompt': 'cyberpunk city', 'model': 'flux.2-pro', 'guidance': 10.0}
    """
    base = dict(base_params) if base_params else {}
    return [{**base, param_name: v} for v in values]


def to_dataframe(**filters: Any) -> Any:
    """Export experiments to a pandas DataFrame.

    Requires pandas to be installed. Useful for data analysis and
    integration with Jupyter notebooks.

    Args:
        **filters: Optional filter conditions (same as find()).

    Returns:
        pandas.DataFrame: DataFrame with experiment data.

    Raises:
        MissingDependencyError: If pandas is not installed.

    Example:
        >>> df = gallium.to_dataframe()
        >>> df.head()
        >>> # With filters
        >>> df = gallium.to_dataframe(model="flux.2-pro")
    """
    return _get_default_tracker().to_dataframe(**filters)
