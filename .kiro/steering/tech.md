# Technology Stack

## Build System & Package Management

- **Package Manager**: `uv` (modern Python package manager)
- **Build Backend**: `uv_build`
- **Python Version**: 3.13+ (strict requirement)
- **Project Structure**: Standard Python package with `src/` layout

## Core Dependencies

- **Speech Recognition**: `faster-whisper` (local Whisper implementation)
- **Audio Processing**: `pyaudio` for microphone input
- **Voice Activity Detection**: `webrtcvad`
- **System Monitoring**: `psutil` for performance metrics
- **ML Framework**: `torch` (PyTorch) for GPU acceleration

## Development Tools

- **Testing**: `pytest` with async support (`pytest-asyncio`)
- **Coverage**: `pytest-cov` for test coverage reporting
- **Type Checking**: `mypy` with strict mode enabled
- **Linting**: `ruff` (replaces flake8, black, isort)
- **Security**: `bandit` for security linting
- **Documentation**: `docformatter` for docstring formatting

## Development Methodology

### Test-Driven Development (TDD)

**Required for all new features**: Write tests first, implement second (Red-Green-Refactor)

### Coverage Standards

- **New code**: 90% minimum line coverage
- **Critical paths**: 100% coverage for core functionality
- **Dead code monitoring**: Weekly `vulture` analysis and cleanup
- **Quality checks**: Run linting/formatting before commits, type checking daily

## Common Commands

### Development Setup

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e .[dev]
```

### Testing

```bash
# Run all tests in parallel (faster execution)
pytest -n auto

# Run with coverage (required for all development)
pytest --cov=src --cov-report=html --cov-fail-under=90

# Run with coverage in parallel (combines coverage from all workers)
pytest -n auto --cov=src --cov-report=html --cov-fail-under=90

# Run test categories in parallel
pytest -n auto -m unit            # Unit tests in parallel
pytest -n auto -m integration     # Integration tests in parallel

pytest -m performance             # Performance benchmarks not in parallel

# TDD workflow
pytest -x --cov=src                         # Stop on first failure
pytest --cov=src --cov-report=term-missing  # Show missing coverage lines
pytest -n auto -x                           # Parallel with stop on first failure

# Recommended parallel workflow
pytest -n auto -m unit && pytest -n auto -m integration  # Run unit tests first, then integration
```

### Code Quality

```bash
# Run type checking
mypy src tests

# Run linting and formatting
ruff check src tests
ruff format src tests

# Run security checks
bandit -r src

# Find dead code
vulture src tests

# Format docstrings
docformatter --in-place --recursive src

# Run all quality checks (recommended before commits)
mypy src tests && ruff check src tests && ruff format src tests && bandit -r src
```

### Periodic Quality Maintenance

**Run these checks regularly to maintain code quality:**

- **Before each commit**: `ruff check` and `ruff format` to catch issues early
- **Daily development**: `mypy` type checking to ensure type safety
- **Weekly**: `vulture` to find and remove dead code
- **Weekly**: Full quality suite including `bandit` security checks
- **Before releases**: Complete quality validation with coverage reports

### Application Usage

```bash
# Run the CLI application
local-ai                    # After installation
python -m local_ai.main     # Module approach

# CLI options
local-ai --verbose                      # Enable debug logging
local-ai --reset-model-cache           # Clear model cache
local-ai --reset-optimization-cache    # Clear optimization cache
```

## Configuration Standards

- **Line Length**: 90 characters (ruff configuration)
- **Target Python**: 3.13
- **Import Sorting**: Handled by ruff
- **Type Hints**: Strict typing required (mypy strict mode)
- **Docstrings**: Google style docstrings required
