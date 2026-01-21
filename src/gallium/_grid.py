"""Grid building for image galleries."""

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from ._imaging import _check_pillow, create_grid_streaming

if TYPE_CHECKING:
    from PIL import Image


# Default maximum images to prevent memory exhaustion
DEFAULT_MAX_IMAGES = 100


def grid(
    images: Sequence[Any],
    cols: int = 3,
    *,
    padding: int = 10,
    background: str = "#ffffff",
    max_size: Optional[int] = None,
    max_images: Optional[int] = DEFAULT_MAX_IMAGES,
    labels: Optional[Sequence[str]] = None,
    label_font_size: int = 14,
    cell_size: Optional[int] = None,
) -> "Image.Image":
    """Combine images into a grid layout.

    Creates a grid of images with optional labels, padding, and thumbnailing.
    Uses streaming to minimize memory usage (loads one image at a time).

    Args:
        images: Sequence of images. Can be:
            - File paths (str or Path)
            - PIL.Image.Image objects
            - bytes (raw image data)
            - Experiment objects (uses .path attribute)
        cols: Number of columns in the grid.
        padding: Padding between images in pixels.
        background: Background color (hex or name like "#ffffff" or "white").
        max_size: If set, thumbnail each image to this max dimension before assembly.
                 Helps prevent memory issues with large images.
        max_images: Maximum number of images to include (default: 100).
                   Set to None to disable the limit (use with caution).
        labels: Optional labels to display below each image.
        label_font_size: Font size for labels in pixels.
        cell_size: Fixed cell size for uniform grid. If None, uses max_size or 256.

    Returns:
        PIL.Image.Image: The combined grid image.

    Raises:
        MissingDependencyError: If Pillow is not installed.
        ValueError: If images is empty or exceeds max_images limit.
        FileNotFoundError: If an image file path doesn't exist.

    Example:
        >>> # Basic grid from file paths
        >>> grid_img = gallium.grid(["img1.png", "img2.png", "img3.png"], cols=3)
        >>> grid_img.save("grid.png")

        >>> # Grid with labels
        >>> grid_img = gallium.grid(
        ...     images,
        ...     cols=2,
        ...     labels=["seed=1", "seed=2", "seed=3", "seed=4"]
        ... )

        >>> # Grid from experiments with thumbnailing
        >>> experiments = gallium.find(prompt__contains="cat")
        >>> grid_img = gallium.grid(
        ...     experiments,
        ...     cols=3,
        ...     max_size=512,
        ...     labels=[f"seed={e.seed}" for e in experiments]
        ... )

        >>> # Disable limit for large grids (use with caution)
        >>> grid_img = gallium.grid(many_images, max_images=None)
    """
    _check_pillow()

    if not images:
        raise ValueError("Cannot create grid from empty image list")

    if max_images is not None and len(images) > max_images:
        raise ValueError(
            f"Too many images ({len(images)}). Maximum is {max_images}. "
            f"Set max_images=None to disable this limit (use with caution)."
        )

    # Convert to list for streaming function
    sources = list(images)
    labels_list = list(labels) if labels else None

    # Determine cell size
    cell_dim = cell_size or max_size

    return create_grid_streaming(
        sources,
        cols=cols,
        padding=padding,
        background=background,
        labels=labels_list,
        label_font_size=label_font_size,
        max_size=max_size,
        cell_width=cell_dim,
        cell_height=cell_dim,
    )


def matrix_grid(
    experiments: Sequence[Any],
    rows: str,
    cols: str,
    *,
    padding: int = 10,
    background: str = "#ffffff",
    max_size: int = 256,
    show_labels: bool = True,
    label_font_size: int = 12,
) -> "Image.Image":
    """Create a matrix grid organized by two experiment attributes.

    Useful for comparing variations across two dimensions (e.g., model vs seed).

    Args:
        experiments: Sequence of Experiment objects.
        rows: Attribute name for row grouping (e.g., "model", "seed").
        cols: Attribute name for column grouping (e.g., "seed", "guidance").
        padding: Padding between images in pixels.
        background: Background color.
        max_size: Maximum dimension for each thumbnail.
        show_labels: Whether to show row/column headers.
        label_font_size: Font size for headers.

    Returns:
        PIL.Image.Image: The matrix grid image.

    Raises:
        ValueError: If experiments is empty or attributes don't exist.
        MissingDependencyError: If Pillow is not installed.

    Example:
        >>> # Compare models across seeds
        >>> experiments = gallium.find(prompt__contains="cat")
        >>> matrix = gallium.matrix_grid(experiments, rows="model", cols="seed")
        >>> matrix.save("model_vs_seed.png")

        >>> # Compare guidance values across seeds
        >>> matrix = gallium.matrix_grid(
        ...     experiments,
        ...     rows="seed",
        ...     cols="model",
        ...     max_size=512
        ... )
    """
    _check_pillow()
    from PIL import Image, ImageDraw

    if not experiments:
        raise ValueError("Cannot create matrix grid from empty experiment list")

    # Extract unique row and column values
    row_values: list[Any] = []
    col_values: list[Any] = []
    exp_map: dict[tuple[Any, Any], Any] = {}

    for exp in experiments:
        row_val = getattr(exp, rows, None)
        col_val = getattr(exp, cols, None)

        if row_val is not None and row_val not in row_values:
            row_values.append(row_val)
        if col_val is not None and col_val not in col_values:
            col_values.append(col_val)

        # Store experiment by (row, col) - last one wins if duplicates
        if row_val is not None and col_val is not None:
            exp_map[(row_val, col_val)] = exp

    if not row_values or not col_values:
        raise ValueError(
            f"No valid values found for rows='{rows}' or cols='{cols}'. "
            f"Ensure experiments have these attributes."
        )

    # Sort values for consistent ordering
    with contextlib.suppress(TypeError):
        row_values = sorted(row_values)
    with contextlib.suppress(TypeError):
        col_values = sorted(col_values)

    # Calculate dimensions
    num_rows = len(row_values)
    num_cols = len(col_values)
    cell_size = max_size

    # Header dimensions
    header_height = (label_font_size + 20) if show_labels else 0
    row_header_width = 100 if show_labels else 0

    # Canvas dimensions
    canvas_width = row_header_width + num_cols * (cell_size + padding) + padding
    canvas_height = header_height + num_rows * (cell_size + padding) + padding

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(canvas)

    # Draw column headers
    if show_labels:
        for col_idx, col_val in enumerate(col_values):
            x = row_header_width + padding + col_idx * (cell_size + padding) + cell_size // 2
            y = header_height // 2
            draw.text((x, y), str(col_val), fill="#000000", anchor="mm")

    # Draw rows
    for row_idx, row_val in enumerate(row_values):
        y_base = header_height + padding + row_idx * (cell_size + padding)

        # Draw row header
        if show_labels:
            draw.text(
                (row_header_width // 2, y_base + cell_size // 2),
                str(row_val),
                fill="#000000",
                anchor="mm",
            )

        # Draw cells
        for col_idx, col_val in enumerate(col_values):
            x_base = row_header_width + padding + col_idx * (cell_size + padding)

            exp = exp_map.get((row_val, col_val))
            if exp is not None and getattr(exp, "path", None):
                try:
                    from ._imaging import load_image, thumbnail

                    img = load_image(exp)
                    img = thumbnail(img, max_size)

                    # Center in cell
                    x_offset = (cell_size - img.width) // 2
                    y_offset = (cell_size - img.height) // 2

                    canvas.paste(img, (x_base + x_offset, y_base + y_offset))
                    img.close()
                except (FileNotFoundError, ValueError):
                    # Draw placeholder for missing image
                    draw.rectangle(
                        [x_base, y_base, x_base + cell_size, y_base + cell_size],
                        outline="#cccccc",
                    )
                    draw.text(
                        (x_base + cell_size // 2, y_base + cell_size // 2),
                        "?",
                        fill="#cccccc",
                        anchor="mm",
                    )
            else:
                # Draw empty cell placeholder
                draw.rectangle(
                    [x_base, y_base, x_base + cell_size, y_base + cell_size],
                    outline="#eeeeee",
                )

    return canvas
