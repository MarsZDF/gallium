"""A/B comparison helper for images."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ._imaging import _check_pillow, create_grid, load_image

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class CompareResult:
    """Result of comparing two images.

    Holds both images and provides convenience methods for
    creating side-by-side comparisons and saving results.

    Attributes:
        a: First image (PIL.Image.Image).
        b: Second image (PIL.Image.Image).
        label_a: Optional label for the first image.
        label_b: Optional label for the second image.

    Example:
        >>> result = gallium.compare("v1.png", "v2.png", labels=["baseline", "modified"])
        >>> result.grid().show()  # Display side-by-side
        >>> result.save("comparison.png")  # Save to file
    """

    a: "Image.Image"
    b: "Image.Image"
    label_a: Optional[str] = None
    label_b: Optional[str] = None

    def grid(self, padding: int = 10, background: str = "#ffffff") -> "Image.Image":
        """Create a side-by-side comparison grid.

        Args:
            padding: Padding between images in pixels.
            background: Background color (hex or name).

        Returns:
            PIL.Image.Image: Side-by-side comparison image.

        Example:
            >>> result = gallium.compare("a.png", "b.png")
            >>> comparison = result.grid(padding=20)
            >>> comparison.show()
        """
        _check_pillow()

        labels = None
        if self.label_a is not None or self.label_b is not None:
            labels = [self.label_a or "", self.label_b or ""]

        return create_grid(
            [self.a, self.b],
            cols=2,
            padding=padding,
            background=background,
            labels=labels,
        )

    def save(
        self,
        path: str,
        *,
        padding: int = 10,
        background: str = "#ffffff",
    ) -> None:
        """Save the side-by-side comparison to a file.

        Args:
            path: Output file path.
            padding: Padding between images in pixels.
            background: Background color (hex or name).

        Example:
            >>> result = gallium.compare("a.png", "b.png")
            >>> result.save("comparison.png")
        """
        grid_img = self.grid(padding=padding, background=background)
        grid_img.save(path)


def compare(
    image_a: Any,
    image_b: Any,
    *,
    labels: Optional[tuple[str, str]] = None,
) -> CompareResult:
    """Compare two images for A/B testing.

    Creates a CompareResult object that holds both images and provides
    methods for creating side-by-side comparisons.

    Args:
        image_a: First image. Can be:
            - File path (str or Path)
            - PIL.Image.Image object
            - bytes (raw image data)
            - Experiment object (uses .path attribute)
        image_b: Second image (same types as image_a).
        labels: Optional tuple of (label_a, label_b) for labeling images.

    Returns:
        CompareResult: Object containing both images with comparison methods.

    Raises:
        MissingDependencyError: If Pillow is not installed.
        FileNotFoundError: If an image file path doesn't exist.

    Example:
        >>> # Basic comparison
        >>> result = gallium.compare("baseline.png", "modified.png")
        >>> result.grid().show()

        >>> # With labels
        >>> result = gallium.compare(
        ...     "v1.png",
        ...     "v2.png",
        ...     labels=("baseline", "after tuning")
        ... )
        >>> result.save("comparison.png")

        >>> # From experiments
        >>> exp_a = gallium.find(seed=42)[0]
        >>> exp_b = gallium.find(seed=123)[0]
        >>> result = gallium.compare(exp_a, exp_b)
    """
    _check_pillow()

    img_a = load_image(image_a)
    img_b = load_image(image_b)

    label_a = labels[0] if labels else None
    label_b = labels[1] if labels else None

    return CompareResult(
        a=img_a,
        b=img_b,
        label_a=label_a,
        label_b=label_b,
    )
