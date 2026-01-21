"""Tests for A/B comparison functionality."""

from pathlib import Path

import pytest

# Skip all tests in this module if Pillow is not installed
pytest.importorskip("PIL")

from PIL import Image

import gallium
from gallium import CompareResult


def create_test_image(width: int = 100, height: int = 100, color: str = "red") -> Image.Image:
    """Create a test image."""
    return Image.new("RGB", (width, height), color)


def save_test_image(path: Path, width: int = 100, height: int = 100, color: str = "red") -> Path:
    """Create and save a test image."""
    img = create_test_image(width, height, color)
    img.save(path)
    return path


class TestCompare:
    """Tests for the compare function."""

    def test_compare_pil_images(self):
        """Test comparing PIL images."""
        img_a = create_test_image(100, 100, "red")
        img_b = create_test_image(100, 100, "blue")

        result = gallium.compare(img_a, img_b)

        assert isinstance(result, CompareResult)
        assert result.a == img_a
        assert result.b == img_b

    def test_compare_file_paths(self, tmp_path):
        """Test comparing images from file paths."""
        path_a = save_test_image(tmp_path / "a.png", color="red")
        path_b = save_test_image(tmp_path / "b.png", color="blue")

        result = gallium.compare(str(path_a), str(path_b))

        assert isinstance(result, CompareResult)
        assert isinstance(result.a, Image.Image)
        assert isinstance(result.b, Image.Image)

    def test_compare_with_labels(self):
        """Test comparing with labels."""
        img_a = create_test_image(100, 100, "red")
        img_b = create_test_image(100, 100, "blue")

        result = gallium.compare(
            img_a, img_b,
            labels=("baseline", "modified")
        )

        assert result.label_a == "baseline"
        assert result.label_b == "modified"

    def test_compare_file_not_found_raises(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            gallium.compare(
                str(tmp_path / "nonexistent.png"),
                str(tmp_path / "also_nonexistent.png")
            )


class TestCompareResult:
    """Tests for the CompareResult class."""

    def test_grid(self):
        """Test creating a grid from CompareResult."""
        img_a = create_test_image(100, 100, "red")
        img_b = create_test_image(100, 100, "blue")

        result = gallium.compare(img_a, img_b)
        grid = result.grid()

        assert isinstance(grid, Image.Image)
        # Grid should be 2 images wide (with padding)
        assert grid.width > 100

    def test_grid_with_labels(self):
        """Test grid with labels."""
        img_a = create_test_image(100, 100, "red")
        img_b = create_test_image(100, 100, "blue")

        result = gallium.compare(
            img_a, img_b,
            labels=("A", "B")
        )
        grid = result.grid()

        assert isinstance(grid, Image.Image)
        # Grid with labels should be taller
        assert grid.height > 100

    def test_grid_custom_padding(self):
        """Test grid with custom padding."""
        img_a = create_test_image(100, 100)
        img_b = create_test_image(100, 100)

        result = gallium.compare(img_a, img_b)

        grid_small = result.grid(padding=5)
        grid_large = result.grid(padding=50)

        assert grid_large.width > grid_small.width

    def test_save(self, tmp_path):
        """Test saving comparison to file."""
        img_a = create_test_image(100, 100, "red")
        img_b = create_test_image(100, 100, "blue")

        result = gallium.compare(img_a, img_b)

        output_path = tmp_path / "comparison.png"
        result.save(str(output_path))

        assert output_path.exists()

        # Verify it's a valid image
        saved_img = Image.open(output_path)
        assert saved_img.width > 100

    def test_save_with_options(self, tmp_path):
        """Test saving with custom options."""
        img_a = create_test_image(100, 100, "red")
        img_b = create_test_image(100, 100, "blue")

        result = gallium.compare(
            img_a, img_b,
            labels=("Baseline", "Modified")
        )

        output_path = tmp_path / "comparison.png"
        result.save(str(output_path), padding=20, background="#cccccc")

        assert output_path.exists()
