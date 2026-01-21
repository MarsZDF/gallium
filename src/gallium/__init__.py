"""Elemental Gallium: Image gallery grids, experiment tracking, and A/B comparison.

Zero-dependency Python library for image generation workflows.
Built for the BFL FLUX.2 Hackathon, works with any image generation tool.

Modules:
    track: SQLite-backed experiment logging
    grid: Combine images into grids (requires Pillow)
    compare: A/B comparison helper (requires Pillow)

Example:
    >>> import gallium
    >>>
    >>> # Track your generations
    >>> gallium.log(prompt="cyberpunk city", seed=42, path="city.png")
    >>>
    >>> # Find what you made
    >>> experiments = gallium.find(prompt__contains="cyberpunk")
    >>>
    >>> # Build a grid for comparison
    >>> grid_img = gallium.grid(experiments, cols=2)
    >>> grid_img.save("comparison_grid.png")
"""

__version__ = "0.9.0"

# Public API - Tracking
# Public API - Comparison
from ._compare import CompareResult, compare

# Public API - Grid
from ._grid import grid, matrix_grid
from ._track import (
    Tracker,
    annotate,
    delete,
    export,
    find,
    init,
    log,
    log_many,
    recent,
    star,
    sweep,
    to_dataframe,
)

# Public API - Types
from ._types import Experiment

# Public API - Exceptions
from .exceptions import (
    GalliumError,
    InvalidFilterError,
    MissingDependencyError,
    SerializationError,
)

__all__ = [
    # Version
    "__version__",
    # Tracking
    "init",
    "log",
    "log_many",
    "find",
    "recent",
    "export",
    "delete",
    "star",
    "annotate",
    "sweep",
    "to_dataframe",
    "Tracker",
    # Types
    "Experiment",
    "CompareResult",
    # Grid & Compare
    "grid",
    "matrix_grid",
    "compare",
    # Exceptions
    "GalliumError",
    "MissingDependencyError",
    "InvalidFilterError",
    "SerializationError",
]
