"""Tests for grid building functionality."""

import io
from pathlib import Path

import pytest

# Skip all tests in this module if Pillow is not installed
pytest.importorskip("PIL")

from PIL import Image

import gallium


def create_test_image(width: int = 100, height: int = 100, color: str = "red") -> Image.Image:
    """Create a test image."""
    return Image.new("RGB", (width, height), color)


def save_test_image(path: Path, width: int = 100, height: int = 100, color: str = "red") -> Path:
    """Create and save a test image."""
    img = create_test_image(width, height, color)
    img.save(path)
    return path


class TestGrid:
    """Tests for the grid function."""

    def test_grid_from_pil_images(self):
        """Test creating grid from PIL images."""
        images = [
            create_test_image(100, 100, "red"),
            create_test_image(100, 100, "green"),
            create_test_image(100, 100, "blue"),
        ]

        result = gallium.grid(images, cols=3)

        assert isinstance(result, Image.Image)
        # Grid should be wider than a single image
        assert result.width > 100

    def test_grid_from_file_paths(self, tmp_path):
        """Test creating grid from file paths."""
        paths = [
            save_test_image(tmp_path / "img1.png", color="red"),
            save_test_image(tmp_path / "img2.png", color="green"),
        ]

        result = gallium.grid([str(p) for p in paths], cols=2)

        assert isinstance(result, Image.Image)

    def test_grid_from_path_objects(self, tmp_path):
        """Test creating grid from Path objects."""
        paths = [
            save_test_image(tmp_path / "img1.png"),
            save_test_image(tmp_path / "img2.png"),
        ]

        result = gallium.grid(paths, cols=2)

        assert isinstance(result, Image.Image)

    def test_grid_from_bytes(self):
        """Test creating grid from bytes."""
        images_bytes = []
        for color in ["red", "green", "blue"]:
            img = create_test_image(100, 100, color)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images_bytes.append(buffer.getvalue())

        result = gallium.grid(images_bytes, cols=3)

        assert isinstance(result, Image.Image)

    def test_grid_with_labels(self):
        """Test grid with labels."""
        images = [
            create_test_image(100, 100, "red"),
            create_test_image(100, 100, "green"),
        ]

        result = gallium.grid(
            images,
            cols=2,
            labels=["Label A", "Label B"]
        )

        assert isinstance(result, Image.Image)
        # Grid should be taller with labels
        assert result.height > 100

    def test_grid_with_padding(self):
        """Test grid with custom padding."""
        images = [
            create_test_image(100, 100),
            create_test_image(100, 100),
        ]

        result_small = gallium.grid(images, cols=2, padding=5)
        result_large = gallium.grid(images, cols=2, padding=50)

        # Larger padding should result in larger grid
        assert result_large.width > result_small.width

    def test_grid_with_max_size(self):
        """Test grid with thumbnailing."""
        images = [
            create_test_image(500, 500),
            create_test_image(500, 500),
        ]

        result = gallium.grid(images, cols=2, max_size=100)

        # Grid should be smaller due to thumbnailing
        assert result.width < 500 * 2

    def test_grid_empty_raises(self):
        """Test that empty image list raises ValueError."""
        with pytest.raises(ValueError):
            gallium.grid([], cols=3)

    def test_grid_max_images_limit(self):
        """Test that max_images raises when exceeded."""
        images = [create_test_image(10, 10) for _ in range(10)]

        # Should work with limit=10
        result = gallium.grid(images, cols=5, max_images=10)
        assert isinstance(result, Image.Image)

        # Should raise with limit=5
        with pytest.raises(ValueError, match="Too many images"):
            gallium.grid(images, cols=5, max_images=5)

    def test_grid_max_images_none_disables_limit(self):
        """Test that max_images=None disables the limit."""
        # Create 150 images (more than default limit of 100)
        images = [create_test_image(10, 10) for _ in range(150)]

        # Should work when limit is disabled
        result = gallium.grid(images, cols=15, max_images=None)
        assert isinstance(result, Image.Image)

    def test_grid_file_not_found_raises(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            gallium.grid([str(tmp_path / "nonexistent.png")], cols=1)

    def test_grid_with_background_color(self):
        """Test grid with custom background."""
        images = [
            create_test_image(100, 100, "red"),
            create_test_image(100, 100, "blue"),
        ]

        result = gallium.grid(images, cols=2, background="#000000")

        assert isinstance(result, Image.Image)
        # Check that a corner pixel is black (background)
        corner_pixel = result.getpixel((0, 0))
        assert corner_pixel == (0, 0, 0)

    def test_grid_single_image(self):
        """Test grid with a single image."""
        images = [create_test_image(100, 100)]

        result = gallium.grid(images, cols=1)

        assert isinstance(result, Image.Image)

    def test_grid_different_sized_images(self):
        """Test grid with images of different sizes."""
        images = [
            create_test_image(100, 100),
            create_test_image(200, 150),
            create_test_image(50, 75),
        ]

        result = gallium.grid(images, cols=3)

        assert isinstance(result, Image.Image)


class TestMatrixGrid:
    """Tests for the matrix_grid function."""

    def test_matrix_grid_basic(self, tmp_path):
        """Test basic matrix grid with experiments."""
        # Create mock experiments with different models and seeds
        experiments = []
        for model in ["flux", "sd"]:
            for seed in [1, 2]:
                path = tmp_path / f"{model}_{seed}.png"
                save_test_image(path, color="red" if model == "flux" else "blue")
                exp = gallium.Experiment(
                    id=len(experiments) + 1,
                    prompt="test",
                    seed=seed,
                    path=str(path),
                    model=model,
                )
                experiments.append(exp)

        result = gallium.matrix_grid(experiments, rows="model", cols="seed")

        assert isinstance(result, Image.Image)
        # Should have space for headers and 2x2 grid
        assert result.width > 0
        assert result.height > 0

    def test_matrix_grid_empty_raises(self):
        """Test that empty experiment list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create matrix grid from empty"):
            gallium.matrix_grid([], rows="model", cols="seed")

    def test_matrix_grid_missing_attribute(self, tmp_path):
        """Test matrix grid with experiments missing the attribute."""
        path = tmp_path / "test.png"
        save_test_image(path)
        exp = gallium.Experiment(
            id=1,
            prompt="test",
            seed=None,  # Missing seed
            path=str(path),
            model=None,  # Missing model
        )

        with pytest.raises(ValueError, match="No valid values found"):
            gallium.matrix_grid([exp], rows="model", cols="seed")

    def test_matrix_grid_without_labels(self, tmp_path):
        """Test matrix grid with labels disabled."""
        experiments = []
        for model in ["flux", "sd"]:
            path = tmp_path / f"{model}.png"
            save_test_image(path)
            exp = gallium.Experiment(
                id=len(experiments) + 1,
                prompt="test",
                seed=1,
                path=str(path),
                model=model,
            )
            experiments.append(exp)

        result = gallium.matrix_grid(
            experiments, rows="model", cols="seed", show_labels=False
        )

        assert isinstance(result, Image.Image)

    def test_matrix_grid_custom_max_size(self, tmp_path):
        """Test matrix grid with custom thumbnail size."""
        path = tmp_path / "test.png"
        save_test_image(path, width=500, height=500)
        exp = gallium.Experiment(
            id=1,
            prompt="test",
            seed=1,
            path=str(path),
            model="flux",
        )

        result = gallium.matrix_grid([exp], rows="model", cols="seed", max_size=100)

        assert isinstance(result, Image.Image)

    def test_matrix_grid_missing_image(self, tmp_path):
        """Test matrix grid handles missing image files gracefully."""
        # Create experiment with non-existent path
        exp = gallium.Experiment(
            id=1,
            prompt="test",
            seed=1,
            path=str(tmp_path / "nonexistent.png"),
            model="flux",
        )

        # Should not raise, should draw placeholder
        result = gallium.matrix_grid([exp], rows="model", cols="seed")
        assert isinstance(result, Image.Image)

    def test_matrix_grid_sparse_matrix(self, tmp_path):
        """Test matrix grid with sparse data (not all cells filled)."""
        experiments = []
        # Only create 2 experiments for a potential 2x2 grid
        for model, seed in [("flux", 1), ("sd", 2)]:
            path = tmp_path / f"{model}_{seed}.png"
            save_test_image(path)
            exp = gallium.Experiment(
                id=len(experiments) + 1,
                prompt="test",
                seed=seed,
                path=str(path),
                model=model,
            )
            experiments.append(exp)

        result = gallium.matrix_grid(experiments, rows="model", cols="seed")

        assert isinstance(result, Image.Image)

    def test_matrix_grid_no_path(self, tmp_path):
        """Test matrix grid with experiment that has no path."""
        exp = gallium.Experiment(
            id=1,
            prompt="test",
            seed=1,
            path=None,  # No path
            model="flux",
        )

        # Should draw empty cell placeholder
        result = gallium.matrix_grid([exp], rows="model", cols="seed")
        assert isinstance(result, Image.Image)
