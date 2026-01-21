"""Data types for gallium experiment tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class Experiment:
    """A tracked image generation experiment.

    Attributes:
        id: Unique identifier for the experiment.
        prompt: The text prompt used for generation.
        seed: Random seed used for generation.
        path: File path to the generated image.
        model: Name of the model used (e.g., "flux.2-pro").
        width: Image width in pixels.
        height: Image height in pixels.
        duration_ms: Generation time in milliseconds.
        image_hash: SHA-256 hash of the image file.
        params: Additional parameters as a JSON-serializable dict.
        starred: Whether this experiment is marked as a favorite.
        notes: Optional notes about the experiment.
        created_at: Timestamp when the experiment was logged.

    Example:
        >>> exp = gallium.find(seed=42)[0]
        >>> print(f"Prompt: {exp.prompt}")
        >>> print(f"Created: {exp.created_at}")
        >>> img = exp.load_image()
        >>> img.show()
    """

    id: int
    prompt: str
    seed: Optional[int] = None
    path: Optional[str] = None
    model: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None
    image_hash: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    starred: bool = False
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def load_image(self) -> "Image.Image":
        """Load the image from path.

        Returns:
            PIL.Image.Image: The loaded image.

        Raises:
            FileNotFoundError: If path is None or the file doesn't exist.
            ValueError: If path is not a regular file.
            MissingDependencyError: If Pillow is not installed.

        Example:
            >>> exp = gallium.find(prompt__contains="cat")[0]
            >>> img = exp.load_image()
            >>> img.show()
        """
        from pathlib import Path

        from .exceptions import MissingDependencyError

        if self.path is None:
            raise FileNotFoundError("Experiment has no associated image path")

        # Validate path exists and is a regular file
        resolved = Path(self.path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        if not resolved.is_file():
            raise ValueError(f"Path is not a regular file: {self.path}")

        try:
            from PIL import Image
        except ImportError as e:
            raise MissingDependencyError(
                "Pillow is required to load images. Install it with: pip install pillow"
            ) from e

        return Image.open(resolved)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        parts = [f"id={self.id}", f"prompt={self.prompt!r}"]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        if self.model is not None:
            parts.append(f"model={self.model!r}")
        return f"Experiment({', '.join(parts)})"
