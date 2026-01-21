"""Tests for FLUX.2-specific helpers."""

import pytest

import gallium.flux as gf


class TestFluxConstants:
    """Tests for FLUX.2 constants."""

    def test_model_constants(self):
        """Test model constant values."""
        assert gf.FLUX_MAX == "flux.2-max"
        assert gf.FLUX_PRO == "flux.2-pro"
        assert gf.FLUX_FLEX == "flux.2-flex"

    def test_models_tuple(self):
        """Test MODELS contains all variants."""
        assert len(gf.MODELS) == 3
        assert gf.FLUX_MAX in gf.MODELS
        assert gf.FLUX_PRO in gf.MODELS
        assert gf.FLUX_FLEX in gf.MODELS

    def test_guidance_defaults(self):
        """Test guidance scale defaults."""
        assert gf.GUIDANCE_RANGE == (1.0, 20.0)
        assert gf.GUIDANCE_DEFAULT == 7.5

    def test_steps_defaults(self):
        """Test inference steps defaults."""
        assert gf.STEPS_RANGE == (20, 50)
        assert gf.STEPS_DEFAULT == 28

    def test_aspect_ratios(self):
        """Test aspect ratio presets."""
        assert gf.ASPECT_RATIOS["1:1"] == (1024, 1024)
        assert gf.ASPECT_RATIOS["16:9"] == (1344, 768)
        assert gf.ASPECT_RATIOS["9:16"] == (768, 1344)

    def test_default_size(self):
        """Test default resolution."""
        assert gf.DEFAULT_SIZE == (1024, 1024)


class TestSeedSweep:
    """Tests for seed_sweep helper."""

    def test_seed_sweep_basic(self):
        """Test basic seed sweep generation."""
        params = gf.seed_sweep("test prompt", seeds=[1, 2, 3])

        assert len(params) == 3
        assert params[0]["prompt"] == "test prompt"
        assert params[0]["seed"] == 1
        assert params[1]["seed"] == 2
        assert params[2]["seed"] == 3

    def test_seed_sweep_default_model(self):
        """Test seed sweep uses flux.2-pro by default."""
        params = gf.seed_sweep("test", seeds=[42])

        assert params[0]["model"] == "flux.2-pro"

    def test_seed_sweep_custom_model(self):
        """Test seed sweep with custom model."""
        params = gf.seed_sweep("test", seeds=[42], model=gf.FLUX_MAX)

        assert params[0]["model"] == "flux.2-max"

    def test_seed_sweep_includes_params(self):
        """Test seed sweep includes guidance and steps in params."""
        params = gf.seed_sweep("test", seeds=[42], guidance=10.0, steps=40)

        assert params[0]["params"]["guidance"] == 10.0
        assert params[0]["params"]["steps"] == 40

    def test_seed_sweep_default_dimensions(self):
        """Test seed sweep uses 1024x1024 by default."""
        params = gf.seed_sweep("test", seeds=[42])

        assert params[0]["width"] == 1024
        assert params[0]["height"] == 1024

    def test_seed_sweep_custom_dimensions(self):
        """Test seed sweep with custom dimensions."""
        params = gf.seed_sweep("test", seeds=[42], width=512, height=768)

        assert params[0]["width"] == 512
        assert params[0]["height"] == 768

    def test_seed_sweep_extra_params(self):
        """Test seed sweep passes through extra params."""
        params = gf.seed_sweep("test", seeds=[42], sampler="euler")

        assert params[0]["params"]["sampler"] == "euler"


class TestGuidanceSweep:
    """Tests for guidance_sweep helper."""

    def test_guidance_sweep_basic(self):
        """Test basic guidance sweep."""
        params = gf.guidance_sweep("test", guidance_values=[5.0, 7.5, 10.0])

        assert len(params) == 3
        assert params[0]["params"]["guidance"] == 5.0
        assert params[1]["params"]["guidance"] == 7.5
        assert params[2]["params"]["guidance"] == 10.0

    def test_guidance_sweep_fixed_seed(self):
        """Test guidance sweep uses same seed for all."""
        params = gf.guidance_sweep("test", [5.0, 10.0], seed=123)

        assert params[0]["seed"] == 123
        assert params[1]["seed"] == 123


class TestModelSweep:
    """Tests for model_sweep helper."""

    def test_model_sweep_default_models(self):
        """Test model sweep includes all FLUX.2 models by default."""
        params = gf.model_sweep("test")

        assert len(params) == 3
        models = [p["model"] for p in params]
        assert "flux.2-max" in models
        assert "flux.2-pro" in models
        assert "flux.2-flex" in models

    def test_model_sweep_custom_models(self):
        """Test model sweep with custom model list."""
        params = gf.model_sweep("test", models=[gf.FLUX_PRO, gf.FLUX_FLEX])

        assert len(params) == 2


class TestPromptSweep:
    """Tests for prompt_sweep helper."""

    def test_prompt_sweep_basic(self):
        """Test basic prompt sweep."""
        prompts = ["cat", "dog", "bird"]
        params = gf.prompt_sweep(prompts)

        assert len(params) == 3
        assert params[0]["prompt"] == "cat"
        assert params[1]["prompt"] == "dog"
        assert params[2]["prompt"] == "bird"

    def test_prompt_sweep_fixed_seed(self):
        """Test prompt sweep uses same seed for all."""
        params = gf.prompt_sweep(["a", "b"], seed=999)

        assert params[0]["seed"] == 999
        assert params[1]["seed"] == 999


class TestAspectRatioSweep:
    """Tests for aspect_ratio_sweep helper."""

    def test_aspect_ratio_sweep_default(self):
        """Test aspect ratio sweep with all ratios."""
        params = gf.aspect_ratio_sweep("test")

        assert len(params) == len(gf.ASPECT_RATIOS)

    def test_aspect_ratio_sweep_custom(self):
        """Test aspect ratio sweep with specific ratios."""
        params = gf.aspect_ratio_sweep("test", ratios=["1:1", "16:9"])

        assert len(params) == 2
        assert params[0]["width"] == 1024
        assert params[0]["height"] == 1024
        assert params[1]["width"] == 1344
        assert params[1]["height"] == 768

    def test_aspect_ratio_sweep_invalid_ratio(self):
        """Test aspect ratio sweep raises for invalid ratio."""
        with pytest.raises(ValueError, match="Unknown aspect ratio"):
            gf.aspect_ratio_sweep("test", ratios=["invalid"])

    def test_aspect_ratio_in_params(self):
        """Test aspect ratio is stored in params."""
        params = gf.aspect_ratio_sweep("test", ratios=["16:9"])

        assert params[0]["params"]["aspect_ratio"] == "16:9"


class TestSweepGrid:
    """Tests for sweep_grid helper."""

    def test_sweep_grid_basic(self, tmp_path):
        """Test sweep_grid creates labeled grid."""
        pytest.importorskip("PIL")
        from PIL import Image

        import gallium

        # Create test experiments
        experiments = []
        for seed in [1, 2, 3, 4]:
            path = tmp_path / f"test_{seed}.png"
            img = Image.new("RGB", (100, 100), "red")
            img.save(path)
            exp = gallium.Experiment(
                id=seed,
                prompt="test",
                seed=seed,
                path=str(path),
                model="flux.2-pro",
            )
            experiments.append(exp)

        result = gf.sweep_grid(experiments, "seed", cols=2)

        assert isinstance(result, Image.Image)

    def test_sweep_grid_no_labels(self, tmp_path):
        """Test sweep_grid without labels."""
        pytest.importorskip("PIL")
        from PIL import Image

        import gallium

        path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), "red")
        img.save(path)
        exp = gallium.Experiment(id=1, prompt="test", seed=42, path=str(path))

        result = gf.sweep_grid([exp], "seed", show_value=False)

        assert isinstance(result, Image.Image)
