# Technology Stack

## Stack

- **Package Manager**: `uv` | **Python**: 3.13+ | **Build**: `uv_build`
- **Speech**: `faster-whisper` | **Audio**: `pyaudio` | **VAD**: `webrtcvad`
- **ML**: `torch` (PyTorch) | **Monitoring**: `psutil`
- **Testing**: `pytest` + `pytest-asyncio` + `pytest-cov`
- **Quality**: `mypy` (strict), `ruff` (lint/format), `bandit` (security)

## TDD Mandate

**Write tests first** (Red-Green-Refactor). 90% coverage minimum, 100% for critical paths.

## Commands

### Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv pip install -e .[dev]                         # Install with dev deps
```

### Testing (run `pytest` directly, no `python -m` needed)

```bash
pytest -n 2                                      # Parallel tests
pytest -n 2 --cov=src --cov-fail-under=90       # With coverage
pytest -n 2 -m unit                              # Unit tests only
pytest -n 2 -m integration                       # Integration tests only
pytest -m performance                            # Performance (no parallel)
pytest -x --cov=src --cov-report=term-missing   # TDD: stop on fail, show missing
```

### Quality

```bash
mypy src tests                                   # Type check
ruff check src tests && ruff format src tests    # Lint + format
bandit -r src                                    # Security scan
vulture src tests                                # Find dead code (weekly)
```

### Run App

```bash
local-ai                                         # After install
local-ai --verbose --reset-model-cache          # With options
```

## Standards

Line length: 90 | Type hints: strict | Docstrings: Google style
