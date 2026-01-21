# Contributing to Gallium

Thank you for your interest in contributing to Gallium! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MarsZDF/gallium.git
   cd gallium
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=gallium --cov-report=term-missing

# Run specific test file
pytest tests/test_track.py -v

# Run tests matching a pattern
pytest tests/ -k "test_log" -v
```

## Code Quality

Before submitting a pull request, ensure all checks pass:

```bash
# Run linter
ruff check src/ tests/

# Run type checker
mypy src/ --strict

# Run tests
pytest tests/ -v
```

## Coding Standards

### Zero Dependencies

Gallium follows the elemental-* family philosophy of **zero runtime dependencies**. All functionality must use only the Python standard library.

Optional dependencies (like Pillow for grid operations) are acceptable but must:
- Be clearly documented as optional
- Raise `MissingDependencyError` with helpful installation instructions when missing
- Be listed in `[project.optional-dependencies]`, not `[project.dependencies]`

### Type Hints

- All public APIs must have complete type hints
- Use Python 3.9+ compatible syntax (`Union[X, Y]` not `X | Y`)
- The codebase must pass `mypy --strict`

### Testing

- Maintain 80%+ test coverage
- Add tests for new functionality
- Test edge cases (empty inputs, None values, error conditions)

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(track): add to_dataframe() method for pandas integration
fix(grid): handle empty image list gracefully
docs(readme): update installation instructions
test(compare): add edge cases for A/B comparison
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all checks pass locally
4. Submit a pull request with a clear description
5. Address any review feedback

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Minimal code to reproduce the issue
- Full error traceback

## Questions?

Feel free to open an issue for questions or discussions about potential features.
