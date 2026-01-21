"""FLUX.2-specific helpers for the BFL FLUX.2 Hackathon.

This module provides constants, presets, and helpers specifically designed
for working with Black Forest Labs' FLUX.2 image generation models.

Example:
    >>> import gallium.flux as gf
    >>>
    >>> # Use model constants
    >>> print(gf.FLUX_PRO)
    'flux.2-pro'
    >>>
    >>> # Generate seed sweep parameters
    >>> params = gf.seed_sweep("cyberpunk city", seeds=[42, 123, 456])
    >>> for p in params:
    ...     # Generate with your API, then log
    ...     gallium.log(**p)
"""

from collections.abc import Sequence
from typing import Any, Optional

# =============================================================================
# FLUX.2 Model Constants
# =============================================================================

#: FLUX.2 [max] - Highest quality, slowest generation
FLUX_MAX: str = "flux.2-max"

#: FLUX.2 [pro] - Professional quality, balanced speed
FLUX_PRO: str = "flux.2-pro"

#: FLUX.2 [flex] - Fast generation, good quality
FLUX_FLEX: str = "flux.2-flex"

#: All available FLUX.2 models
MODELS: tuple[str, ...] = (FLUX_MAX, FLUX_PRO, FLUX_FLEX)

# =============================================================================
# Default Parameter Ranges (based on FLUX.2 documentation)
# =============================================================================

#: Recommended guidance scale range
GUIDANCE_RANGE: tuple[float, float] = (1.0, 20.0)

#: Default guidance scale
GUIDANCE_DEFAULT: float = 7.5

#: Recommended inference steps range
STEPS_RANGE: tuple[int, int] = (20, 50)

#: Default inference steps
STEPS_DEFAULT: int = 28

#: Common aspect ratios for FLUX.2
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
}

#: Default resolution
DEFAULT_SIZE: tuple[int, int] = (1024, 1024)


# =============================================================================
# Sweep Helpers
# =============================================================================


def seed_sweep(
    prompt: str,
    seeds: Sequence[int],
    *,
    model: str = FLUX_PRO,
    guidance: float = GUIDANCE_DEFAULT,
    steps: int = STEPS_DEFAULT,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **extra_params: Any,
) -> list[dict[str, Any]]:
    """Generate parameter dicts for a seed sweep.

    Creates a list of parameter dictionaries ready for generation,
    varying only the seed while keeping other parameters constant.

    Args:
        prompt: The text prompt for generation.
        seeds: List of seeds to sweep over.
        model: FLUX.2 model to use (default: flux.2-pro).
        guidance: Guidance scale (default: 7.5).
        steps: Number of inference steps (default: 28).
        width: Image width (default: 1024).
        height: Image height (default: 1024).
        **extra_params: Additional parameters to include.

    Returns:
        List of parameter dicts, one per seed.

    Example:
        >>> params = gf.seed_sweep("cat portrait", seeds=[1, 2, 3, 4])
        >>> for p in params:
        ...     image = your_api.generate(**p)
        ...     image.save(f"cat_seed_{p['seed']}.png")
        ...     gallium.log(**p, path=f"cat_seed_{p['seed']}.png")
    """
    w, h = width or DEFAULT_SIZE[0], height or DEFAULT_SIZE[1]

    return [
        {
            "prompt": prompt,
            "seed": seed,
            "model": model,
            "width": w,
            "height": h,
            "params": {"guidance": guidance, "steps": steps, **extra_params},
        }
        for seed in seeds
    ]


def guidance_sweep(
    prompt: str,
    guidance_values: Sequence[float],
    *,
    seed: int = 42,
    model: str = FLUX_PRO,
    steps: int = STEPS_DEFAULT,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **extra_params: Any,
) -> list[dict[str, Any]]:
    """Generate parameter dicts for a guidance scale sweep.

    Creates a list of parameter dictionaries varying guidance scale
    while keeping seed constant - useful for finding optimal guidance.

    Args:
        prompt: The text prompt for generation.
        guidance_values: List of guidance scales to try.
        seed: Fixed seed for comparison (default: 42).
        model: FLUX.2 model to use (default: flux.2-pro).
        steps: Number of inference steps (default: 28).
        width: Image width (default: 1024).
        height: Image height (default: 1024).
        **extra_params: Additional parameters to include.

    Returns:
        List of parameter dicts, one per guidance value.

    Example:
        >>> params = gf.guidance_sweep("landscape", [5.0, 7.5, 10.0, 15.0])
        >>> for p in params:
        ...     image = your_api.generate(**p)
        ...     guidance = p['params']['guidance']
        ...     image.save(f"landscape_g{guidance}.png")
    """
    w, h = width or DEFAULT_SIZE[0], height or DEFAULT_SIZE[1]

    return [
        {
            "prompt": prompt,
            "seed": seed,
            "model": model,
            "width": w,
            "height": h,
            "params": {"guidance": g, "steps": steps, **extra_params},
        }
        for g in guidance_values
    ]


def model_sweep(
    prompt: str,
    models: Optional[Sequence[str]] = None,
    *,
    seed: int = 42,
    guidance: float = GUIDANCE_DEFAULT,
    steps: int = STEPS_DEFAULT,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **extra_params: Any,
) -> list[dict[str, Any]]:
    """Generate parameter dicts for a model comparison sweep.

    Creates a list of parameter dictionaries varying the model
    while keeping other parameters constant.

    Args:
        prompt: The text prompt for generation.
        models: List of models to compare (default: all FLUX.2 models).
        seed: Fixed seed for comparison (default: 42).
        guidance: Guidance scale (default: 7.5).
        steps: Number of inference steps (default: 28).
        width: Image width (default: 1024).
        height: Image height (default: 1024).
        **extra_params: Additional parameters to include.

    Returns:
        List of parameter dicts, one per model.

    Example:
        >>> params = gf.model_sweep("portrait photo")
        >>> for p in params:
        ...     image = your_api.generate(**p)
        ...     image.save(f"portrait_{p['model']}.png")
    """
    if models is None:
        models = list(MODELS)

    w, h = width or DEFAULT_SIZE[0], height or DEFAULT_SIZE[1]

    return [
        {
            "prompt": prompt,
            "seed": seed,
            "model": model,
            "width": w,
            "height": h,
            "params": {"guidance": guidance, "steps": steps, **extra_params},
        }
        for model in models
    ]


def prompt_sweep(
    prompts: Sequence[str],
    *,
    seed: int = 42,
    model: str = FLUX_PRO,
    guidance: float = GUIDANCE_DEFAULT,
    steps: int = STEPS_DEFAULT,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **extra_params: Any,
) -> list[dict[str, Any]]:
    """Generate parameter dicts for a prompt comparison sweep.

    Creates a list of parameter dictionaries varying the prompt
    while keeping other parameters constant.

    Args:
        prompts: List of prompts to compare.
        seed: Fixed seed for comparison (default: 42).
        model: FLUX.2 model to use (default: flux.2-pro).
        guidance: Guidance scale (default: 7.5).
        steps: Number of inference steps (default: 28).
        width: Image width (default: 1024).
        height: Image height (default: 1024).
        **extra_params: Additional parameters to include.

    Returns:
        List of parameter dicts, one per prompt.

    Example:
        >>> styles = ["photo of a cat", "oil painting of a cat", "sketch of a cat"]
        >>> params = gf.prompt_sweep(styles)
    """
    w, h = width or DEFAULT_SIZE[0], height or DEFAULT_SIZE[1]

    return [
        {
            "prompt": prompt,
            "seed": seed,
            "model": model,
            "width": w,
            "height": h,
            "params": {"guidance": guidance, "steps": steps, **extra_params},
        }
        for prompt in prompts
    ]


def aspect_ratio_sweep(
    prompt: str,
    ratios: Optional[Sequence[str]] = None,
    *,
    seed: int = 42,
    model: str = FLUX_PRO,
    guidance: float = GUIDANCE_DEFAULT,
    steps: int = STEPS_DEFAULT,
    **extra_params: Any,
) -> list[dict[str, Any]]:
    """Generate parameter dicts for an aspect ratio sweep.

    Creates a list of parameter dictionaries varying the aspect ratio
    while keeping other parameters constant.

    Args:
        prompt: The text prompt for generation.
        ratios: List of aspect ratio keys (default: all standard ratios).
                Valid keys: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
        seed: Fixed seed for comparison (default: 42).
        model: FLUX.2 model to use (default: flux.2-pro).
        guidance: Guidance scale (default: 7.5).
        steps: Number of inference steps (default: 28).
        **extra_params: Additional parameters to include.

    Returns:
        List of parameter dicts, one per aspect ratio.

    Example:
        >>> params = gf.aspect_ratio_sweep("landscape photo", ratios=["16:9", "4:3", "1:1"])
    """
    if ratios is None:
        ratios = list(ASPECT_RATIOS.keys())

    result = []
    for ratio in ratios:
        if ratio not in ASPECT_RATIOS:
            raise ValueError(
                f"Unknown aspect ratio: {ratio}. "
                f"Valid ratios: {', '.join(ASPECT_RATIOS.keys())}"
            )
        w, h = ASPECT_RATIOS[ratio]
        result.append({
            "prompt": prompt,
            "seed": seed,
            "model": model,
            "width": w,
            "height": h,
            "params": {
                "guidance": guidance,
                "steps": steps,
                "aspect_ratio": ratio,
                **extra_params,
            },
        })

    return result


# =============================================================================
# Grid Helpers
# =============================================================================


def sweep_grid(
    experiments: Sequence[Any],
    param_name: str,
    *,
    cols: int = 4,
    max_size: int = 256,
    show_value: bool = True,
) -> Any:
    """Create a grid from sweep experiments, labeled by parameter value.

    Convenience function that combines find() results with grid(),
    automatically labeling each cell with the sweep parameter value.

    Args:
        experiments: List of Experiment objects from a sweep.
        param_name: Parameter name to show in labels (e.g., "seed", "guidance").
        cols: Number of columns in grid.
        max_size: Maximum thumbnail dimension.
        show_value: Whether to show parameter value as label.

    Returns:
        PIL.Image.Image: The grid image.

    Example:
        >>> experiments = gallium.find(prompt__contains="cyberpunk")
        >>> grid = gf.sweep_grid(experiments, "seed", cols=4)
        >>> grid.save("seed_comparison.png")
    """
    from . import grid as grid_func

    labels = None
    if show_value:
        labels = []
        for exp in experiments:
            # Try to get value from experiment attribute or params
            value = getattr(exp, param_name, None)
            if value is None and hasattr(exp, "params"):
                value = exp.params.get(param_name)
            labels.append(f"{param_name}={value}" if value is not None else "")

    return grid_func(experiments, cols=cols, max_size=max_size, labels=labels)
