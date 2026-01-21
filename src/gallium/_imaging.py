"""Pillow helpers for image operations."""

import hashlib
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from .exceptions import MissingDependencyError

if TYPE_CHECKING:
    from PIL import Image

# Thread-safe Pillow availability check
_pil_check_lock = threading.Lock()
_pil_available: Optional[bool] = None


def _check_pillow() -> None:
    """Check if Pillow is available, raise if not.

    Thread-safe: uses a lock to prevent race conditions during the initial check.
    """
    global _pil_available

    # Fast path: already checked
    if _pil_available is not None:
        if not _pil_available:
            raise MissingDependencyError(
                "Pillow is required for grid operations. Install it with: pip install pillow"
            )
        return

    # Slow path: need to check (with lock for thread safety)
    with _pil_check_lock:
        # Double-check after acquiring lock
        if _pil_available is not None:
            if not _pil_available:
                raise MissingDependencyError(
                    "Pillow is required for grid operations. Install it with: pip install pillow"
                )
            return

        try:
            import PIL  # noqa: F401

            _pil_available = True
        except ImportError as e:
            _pil_available = False
            raise MissingDependencyError(
                "Pillow is required for grid operations. Install it with: pip install pillow"
            ) from e


def compute_image_hash(path: Union[str, Path]) -> Optional[str]:
    """Compute SHA-256 hash of an image file.

    Args:
        path: Path to the image file.

    Returns:
        str or None: SHA-256 hash prefixed with "sha256:" or None if file doesn't exist.

    Example:
        >>> hash = compute_image_hash("output.png")
        >>> print(hash)
        sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    """
    path = Path(path)

    if not path.exists():
        warnings.warn(f"Image file not found: {path}", UserWarning, stacklevel=2)
        return None

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return f"sha256:{sha256.hexdigest()}"


def _validate_path(path: Path) -> Path:
    """Validate and resolve a file path safely.

    Ensures the path:
    - Is resolved to an absolute path
    - Exists on the filesystem
    - Is a regular file (not a directory or symlink to directory)

    Args:
        path: Path to validate.

    Returns:
        Resolved absolute path.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If path is not a regular file.
    """
    resolved = path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    return resolved


def load_image(source: Any) -> "Image.Image":
    """Load an image from various sources.

    Args:
        source: Can be:
            - str or Path: file path
            - PIL.Image.Image: returned as-is
            - bytes: loaded as image data
            - Experiment: uses .path attribute

    Returns:
        PIL.Image.Image: The loaded image.

    Raises:
        MissingDependencyError: If Pillow is not installed.
        FileNotFoundError: If file path doesn't exist.
        ValueError: If source type is not supported or path is invalid.
    """
    _check_pillow()
    from PIL import Image

    # Already a PIL Image
    if isinstance(source, Image.Image):
        return source

    # Experiment object
    if hasattr(source, "path") and hasattr(source, "prompt"):
        # Duck typing for Experiment
        if source.path is None:
            raise FileNotFoundError("Experiment has no associated image path")
        validated_path = _validate_path(Path(source.path))
        return Image.open(validated_path)

    # File path
    if isinstance(source, (str, Path)):
        validated_path = _validate_path(Path(source))
        return Image.open(validated_path)

    # Bytes
    if isinstance(source, bytes):
        import io

        return Image.open(io.BytesIO(source))

    raise ValueError(
        f"Unsupported image source type: {type(source).__name__}. "
        "Expected str, Path, PIL.Image.Image, bytes, or Experiment."
    )


def thumbnail(image: "Image.Image", max_size: int) -> "Image.Image":
    """Create a thumbnail of an image.

    Args:
        image: PIL Image to thumbnail.
        max_size: Maximum dimension (width or height).

    Returns:
        PIL.Image.Image: Thumbnailed image (may be a copy).
    """
    _check_pillow()

    if image.width <= max_size and image.height <= max_size:
        return image

    # Create a copy to avoid modifying the original
    img = image.copy()
    img.thumbnail((max_size, max_size))
    return img


def create_grid(
    images: list["Image.Image"],
    cols: int,
    padding: int = 10,
    background: str = "#ffffff",
    labels: Optional[list[str]] = None,
    label_font_size: int = 14,
    label_color: str = "#000000",
) -> "Image.Image":
    """Create a grid of images.

    Args:
        images: List of PIL Images.
        cols: Number of columns.
        padding: Padding between images in pixels.
        background: Background color (hex or name).
        labels: Optional labels for each image.
        label_font_size: Font size for labels.
        label_color: Color for label text (hex or name).

    Returns:
        PIL.Image.Image: The combined grid image.
    """
    _check_pillow()
    from PIL import Image, ImageDraw

    if not images:
        raise ValueError("Cannot create grid from empty image list")

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols

    # Find max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Add space for labels if provided
    label_height = label_font_size + 10 if labels else 0

    # Calculate canvas size
    canvas_width = cols * max_width + (cols + 1) * padding
    canvas_height = rows * (max_height + label_height) + (rows + 1) * padding

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(canvas)

    # Place images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = padding + col * (max_width + padding)
        y = padding + row * (max_height + label_height + padding)

        # Center image in cell
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2

        # Paste image
        canvas.paste(img, (x + x_offset, y + y_offset))

        # Draw label if provided
        if labels and idx < len(labels):
            label = labels[idx]
            label_x = x + max_width // 2
            label_y = y + max_height + 5
            draw.text(
                (label_x, label_y),
                label,
                fill=label_color,
                anchor="mt",  # middle-top anchor
            )

    return canvas


def create_grid_streaming(
    sources: list[Any],
    cols: int,
    padding: int = 10,
    background: str = "#ffffff",
    labels: Optional[list[str]] = None,
    label_font_size: int = 14,
    label_color: str = "#000000",
    max_size: Optional[int] = None,
    cell_width: Optional[int] = None,
    cell_height: Optional[int] = None,
) -> "Image.Image":
    """Create a grid by streaming images one at a time (memory efficient).

    Unlike create_grid(), this function loads, processes, and pastes each image
    individually, then releases it. This prevents OOM with large image sets.

    Args:
        sources: List of image sources (paths, bytes, Experiments, etc.).
        cols: Number of columns.
        padding: Padding between images in pixels.
        background: Background color (hex or name).
        labels: Optional labels for each image.
        label_font_size: Font size for labels.
        label_color: Color for label text (hex or name).
        max_size: If set, thumbnail each image to this max dimension.
        cell_width: Fixed cell width. If None, uses max_size or 256.
        cell_height: Fixed cell height. If None, uses max_size or 256.

    Returns:
        PIL.Image.Image: The combined grid image.
    """
    _check_pillow()
    from PIL import Image, ImageDraw

    if not sources:
        raise ValueError("Cannot create grid from empty image list")

    # Determine cell dimensions
    default_size = max_size or 256
    cell_w = cell_width or default_size
    cell_h = cell_height or default_size

    # Calculate grid dimensions
    num_images = len(sources)
    rows = (num_images + cols - 1) // cols

    # Add space for labels if provided
    label_height = label_font_size + 10 if labels else 0

    # Calculate canvas size
    canvas_width = cols * cell_w + (cols + 1) * padding
    canvas_height = rows * (cell_h + label_height) + (rows + 1) * padding

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(canvas)

    # Stream images one at a time
    for idx, source in enumerate(sources):
        # Track if we loaded from file (so we know to close it)
        from PIL import Image as PILImage
        should_close = not isinstance(source, PILImage.Image)

        # Load image
        img = load_image(source)

        # Thumbnail if needed
        if max_size is not None:
            img = thumbnail(img, max_size)

        # Calculate position
        row = idx // cols
        col = idx % cols

        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + label_height + padding)

        # Center image in cell
        x_offset = (cell_w - img.width) // 2
        y_offset = (cell_h - img.height) // 2

        # Paste image
        canvas.paste(img, (x + x_offset, y + y_offset))

        # Close image to free memory (only if we loaded it from file/bytes)
        if should_close:
            img.close()

        # Draw label if provided
        if labels and idx < len(labels):
            label = labels[idx]
            label_x = x + cell_w // 2
            label_y = y + cell_h + 5
            draw.text(
                (label_x, label_y),
                label,
                fill=label_color,
                anchor="mt",
            )

    return canvas
