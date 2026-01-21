# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-01-20

### Added

- **Experiment Tracking** (`gallium.track`):
  - `log()` - Log image generation experiments with full metadata
  - `log_many()` - Batch insert for bulk operations
  - `find()` - Query experiments with flexible filters
  - `recent()` - Get most recent experiments
  - `export()` - Export to CSV or JSON
  - `init()` - Configure custom database path
  - `Tracker` - Class for custom tracker instances

- **Grid Building** (`gallium.grid`):
  - `grid()` - Combine images into grid layouts
  - Support for file paths, PIL Images, bytes, and Experiment objects
  - Labels, padding, background color, and thumbnailing options

- **A/B Comparison** (`gallium.compare`):
  - `compare()` - Compare two images side-by-side
  - `CompareResult` - Result class with `grid()` and `save()` methods

- **Data Types**:
  - `Experiment` - Dataclass for tracked experiments with `load_image()` method
  - `CompareResult` - Dataclass for comparison results

- **Exception Handling**:
  - `GalliumError` - Base exception class
  - `MissingDependencyError` - Raised when Pillow is missing
  - `InvalidFilterError` - Raised for unsupported filters
  - `SerializationError` - Raised for non-JSON-serializable params

### Features

- **Zero runtime dependencies** - Core tracking works with stdlib only
- **Pillow optional** - Required only for grid/compare features
- **Lazy initialization** - Database auto-initializes on first `log()` call
- **Auto-computed hashes** - SHA-256 image hashes computed automatically
- **Type safety** - Full type hints with py.typed marker
- **Python 3.9+** - Compatible with Python 3.9-3.13

### Documentation

- Comprehensive README with examples
- Docstrings with usage examples on all public functions
- Quick start guide for FLUX.2 workflows

[Unreleased]: https://github.com/MarsZDF/gallium/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MarsZDF/gallium/releases/tag/v1.0.0
