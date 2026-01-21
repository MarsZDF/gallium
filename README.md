# Elemental Gallium 🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](https://github.com/MarsZDF/gallium)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarsZDF/gallium/blob/main/demo.ipynb)

**Image gallery grids, experiment tracking, and A/B comparison for image generation workflows.**

Part of the [elemental family](https://github.com/MarsZDF): lightweight, zero-dependency Python libraries.
Originally inspired by the [BFL FLUX.2 Hackathon](https://www.blackforestlabs.ai/) — includes first-class support for FLUX.2 models, Runware, and BFL API integrations.

**Interactive Demos:**
- [General Demo](https://colab.research.google.com/github/MarsZDF/gallium/blob/main/demo.ipynb) - Core features walkthrough
- [FLUX.2 Live Demo](https://colab.research.google.com/github/MarsZDF/gallium/blob/main/flux2_demo.ipynb) - Live image generation with BFL API (requires API key)

---

## Quick Start

```python
import gallium

# Track your generations
gallium.log(prompt="cyberpunk city at night", seed=42, path="city.png", model="flux.2-pro")
gallium.log(prompt="cyberpunk city at night", seed=123, path="city2.png", model="flux.2-pro")
gallium.log(prompt="forest in morning mist", seed=42, path="forest.png", model="flux.2-pro")

# Find what you made
experiments = gallium.find(prompt__contains="cyberpunk")
print(f"Found {len(experiments)} cyberpunk images")

# Build a grid for comparison
grid_image = gallium.grid(experiments, cols=2, labels=[f"seed={e.seed}" for e in experiments])
grid_image.save("comparison_grid.png")
```

---

## Installation

```bash
# Core library (zero dependencies, tracking only)
pip install elemental-gallium

# With grid/compare support (requires Pillow)
pip install elemental-gallium[grid]
```

**Requirements:** Python 3.9+

---

## Features

### Experiment Tracking

Log your image generations with full metadata and query them later.

```python
import gallium

# Log a generation (auto-initializes DB)
gallium.log(
    prompt="cyberpunk city",
    seed=42,
    path="output.png",
    model="flux.2-pro",
    width=1024,
    height=768,
    duration_ms=3420,
    params={"guidance": 7.5, "steps": 50}
)

# Batch insert for bulk operations
gallium.log_many([
    {"prompt": "cat", "seed": 1, "path": "out1.png", "model": "flux.2-pro"},
    {"prompt": "cat", "seed": 2, "path": "out2.png", "model": "flux.2-pro"},
])
```

### Querying

Find experiments with flexible filters.

```python
# By prompt
gallium.find(prompt__contains="cyberpunk")
gallium.find(prompt__startswith="a photo of")

# By metadata
gallium.find(seed=42)
gallium.find(model="flux.2-pro")
gallium.find(width=1024, height=768)

# By date
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)
gallium.find(created_after=yesterday)

# Recent experiments
gallium.recent(10)  # Last 10 experiments
```

### Grid Building

Combine images into grids for comparison. Requires Pillow.

```python
import gallium

# Basic grid from file paths
grid_img = gallium.grid(["img1.png", "img2.png", "img3.png"], cols=3)
grid_img.save("grid.png")

# With labels and styling
grid_img = gallium.grid(
    experiments,
    cols=3,
    padding=10,
    background="#000000",
    labels=[f"seed={e.seed}" for e in experiments],
    label_font_size=14,
    label_color="#ffffff"
)

# Prevent OOM with large images
grid_img = gallium.grid(images, cols=4, max_size=512)  # Thumbnail before assembly
```

**Input flexibility:** accepts file paths, PIL Images, bytes, or Experiment objects.

### A/B Comparison

Compare two images side-by-side.

```python
import gallium

# Compare two images
result = gallium.compare("v1.png", "v2.png")
result.grid().show()        # Display side-by-side
result.save("comparison.png")  # Save to file

# With labels
result = gallium.compare(
    "baseline.png",
    "modified.png",
    labels=("baseline", "after tuning")
)
```

### Export Data

```python
# Export to CSV (default)
gallium.export("csv")  # Creates gallium_export.csv

# Export to JSON
gallium.export("json", path="my_experiments.json")

# Export to HTML gallery
gallium.export("html", path="gallery.html", title="My Experiments")
```

### Star and Annotate

Mark your best experiments for later reference.

```python
# Star an experiment
gallium.star(exp_id)

# Add notes
gallium.annotate(exp_id, "Best composition from seed sweep")

# Find starred experiments
best = gallium.find(starred=True)
```

### Matrix Grid

Compare experiments across two dimensions (e.g., model vs. seed).

```python
experiments = gallium.find(prompt__contains="portrait")
matrix = gallium.matrix_grid(
    experiments,
    rows="model",
    cols="seed",
    max_size=256,
    show_labels=True
)
matrix.save("model_vs_seed_comparison.png")
```

---

## FLUX.2 Integration

The `gallium.flux` module provides constants, presets, and sweep helpers specifically designed for FLUX.2 workflows.

```python
import gallium
import gallium.flux as gf

# Model constants
print(gf.FLUX_MAX)   # "flux.2-max"
print(gf.FLUX_PRO)   # "flux.2-pro"
print(gf.FLUX_FLEX)  # "flux.2-flex"

# Parameter defaults
print(gf.GUIDANCE_DEFAULT)  # 7.5
print(gf.STEPS_DEFAULT)     # 28
print(gf.ASPECT_RATIOS)     # {"1:1": (1024, 1024), "16:9": (1344, 768), ...}
```

### Parameter Sweeps

Generate parameter dictionaries for systematic experimentation.

```python
import gallium.flux as gf

# Seed sweep - find good compositions
params = gf.seed_sweep("cyberpunk city", seeds=[42, 123, 456, 789])
for p in params:
    image = your_api.generate(**p)
    gallium.log(**p, path=f"out_{p['seed']}.png")

# Guidance sweep - tune prompt adherence
params = gf.guidance_sweep("portrait photo", guidance_values=[3.0, 7.5, 12.0, 20.0])

# Model sweep - compare quality tiers
params = gf.model_sweep("landscape scene")  # Tests all FLUX.2 models

# Aspect ratio sweep - find best framing
params = gf.aspect_ratio_sweep("wide landscape", ratios=["16:9", "3:2", "1:1"])
```

### Sweep Grid Helper

Create labeled comparison grids from sweep results.

```python
import gallium.flux as gf

# Generate and log a seed sweep
for p in gf.seed_sweep("night sky", seeds=[1, 2, 3, 4]):
    image = your_api.generate(**p)
    gallium.log(**p, path=f"sky_{p['seed']}.png")

# Create comparison grid with automatic labels
experiments = gallium.find(prompt__contains="night sky")
grid = gf.sweep_grid(experiments, "seed", cols=2, max_size=512)
grid.save("seed_comparison.png")
```

### API Integration Examples

See the `examples/` directory for complete integration examples:

- **[runware_integration.py](examples/runware_integration.py)** - Full Runware API integration with hackathon workflow
- **[bfl_api_integration.py](examples/bfl_api_integration.py)** - BFL API integration with seed/guidance/model sweeps

---

## API Reference

### Tracking Functions

| Function | Description |
|----------|-------------|
| `log(prompt, *, seed, path, model, width, height, duration_ms, image_hash, params, starred, notes)` | Log an experiment |
| `log_many(experiments)` | Batch insert experiments |
| `find(**filters)` | Query experiments |
| `recent(limit=10)` | Get recent experiments |
| `star(id)` | Mark experiment as starred |
| `annotate(id, notes)` | Add notes to experiment |
| `export(format, path, title)` | Export to CSV, JSON, or HTML |
| `init(db_path)` | Set custom database path |

### Grid & Compare Functions

| Function | Description |
|----------|-------------|
| `grid(images, cols, *, padding, background, max_size, labels, label_font_size, label_color)` | Create image grid |
| `matrix_grid(experiments, rows, cols, *, max_size, show_labels, label_font_size, label_color, max_label_length)` | Create matrix comparison grid |
| `compare(image_a, image_b, *, labels)` | Compare two images |

### FLUX.2 Module (`gallium.flux`)

| Constant/Function | Description |
|----------|-------------|
| `FLUX_MAX`, `FLUX_PRO`, `FLUX_FLEX` | Model name constants |
| `MODELS` | Tuple of all FLUX.2 models |
| `GUIDANCE_DEFAULT`, `STEPS_DEFAULT` | Default parameter values |
| `ASPECT_RATIOS` | Dict of aspect ratio to (width, height) |
| `seed_sweep(prompt, seeds, **kwargs)` | Generate seed variation params |
| `guidance_sweep(prompt, guidance_values, **kwargs)` | Generate guidance sweep params |
| `model_sweep(prompt, models, **kwargs)` | Generate model comparison params |
| `prompt_sweep(prompts, **kwargs)` | Generate prompt variation params |
| `aspect_ratio_sweep(prompt, ratios, **kwargs)` | Generate aspect ratio params |
| `sweep_grid(experiments, param_name, **kwargs)` | Create labeled grid from sweep |

### Supported Filters for `find()`

| Filter | Example |
|--------|---------|
| `id` | `find(id=1)` |
| `prompt__contains` | `find(prompt__contains="cat")` |
| `prompt__startswith` | `find(prompt__startswith="a photo")` |
| `prompt` or `prompt__exact` | `find(prompt="exact match")` |
| `seed` | `find(seed=42)` |
| `model` | `find(model="flux.2-pro")` |
| `width`, `height` | `find(width=1024)` |
| `starred` | `find(starred=True)` |
| `notes__contains` | `find(notes__contains="best")` |
| `created_after`, `created_before` | `find(created_after=datetime(...))` |

---

## Data Types

### Experiment

```python
@dataclass
class Experiment:
    id: int
    prompt: str
    seed: int | None
    path: str | None
    model: str | None
    width: int | None
    height: int | None
    duration_ms: int | None
    image_hash: str | None
    params: dict[str, Any]
    starred: bool
    notes: str | None
    created_at: datetime

    def load_image(self) -> Image.Image:
        """Load the image from path."""
```

### CompareResult

```python
@dataclass
class CompareResult:
    a: Image.Image
    b: Image.Image
    label_a: str | None
    label_b: str | None

    def grid(self, padding: int = 10) -> Image.Image: ...
    def save(self, path: str) -> None: ...
```

---

## Database Behavior

- **Auto-initializes** on first `log()` call, zero configuration required
- **Default location:** `./gallium.db` (current working directory)
- **Configurable:** `gallium.init(db_path="custom/path.db")`
- **Safe queries:** `find()` and `recent()` return empty list if DB doesn't exist yet

### Custom Tracker Instance

```python
# For multiple databases or explicit lifecycle management
tracker = gallium.Tracker(db_path="project_a.db")
tracker.log(prompt="test", seed=1)
tracker.find(seed=1)
tracker.close()
```

---

## Error Handling

```python
from gallium import GalliumError, MissingDependencyError, InvalidFilterError, SerializationError

try:
    gallium.grid(images)
except MissingDependencyError:
    print("Install Pillow: pip install pillow")

try:
    gallium.find(invalid_filter="value")
except InvalidFilterError as e:
    print(f"Bad filter: {e}")

try:
    gallium.log(prompt="test", params={"func": lambda x: x})
except SerializationError as e:
    print(f"params must be JSON-serializable: {e}")
```

---

## Integration Example: FLUX.2 Workflow

```python
import gallium
import time
from bfl import flux  # hypothetical FLUX API

def generate_and_track(prompt: str, seed: int) -> str:
    """Generate an image and track the experiment."""
    start = time.time()

    # Generate image
    image = flux.generate(prompt=prompt, seed=seed, model="flux.2-pro")
    path = f"outputs/{seed}.png"
    image.save(path)

    # Track experiment
    gallium.log(
        prompt=prompt,
        seed=seed,
        path=path,
        model="flux.2-pro",
        width=image.width,
        height=image.height,
        duration_ms=int((time.time() - start) * 1000),
        params={"guidance": 7.5}
    )

    return path

# Generate variations
for seed in [42, 123, 456, 789]:
    generate_and_track("cyberpunk city at night", seed)

# Compare results
experiments = gallium.find(prompt__contains="cyberpunk")
grid_img = gallium.grid(
    experiments,
    cols=2,
    labels=[f"seed={e.seed}\n{e.duration_ms}ms" for e in experiments]
)
grid_img.save("cyberpunk_comparison.png")
```

---

## Elemental Family

Gallium is part of a collection of lightweight, zero-dependency Python libraries:

- **[elemental-neon](https://github.com/MarsZDF/neon)** - Near-equality and tolerance arithmetic for floating-point numbers
- **[elemental-rhodium](https://github.com/MarsZDF/rhodium)** - Circular arithmetic for geographic coordinates
- **[elemental-indium](https://github.com/MarsZDF/indium)** - Text inspection and invisible character detection
- **[elemental-xenon](https://github.com/MarsZDF/xenon)** - LLM XML repair and sanitization

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.
