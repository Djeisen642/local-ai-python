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
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Fast unit tests only
pytest -m integration   # Integration tests
pytest -m performance   # Performance benchmarks
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

# Format docstrings
docformatter --in-place --recursive src
```

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
