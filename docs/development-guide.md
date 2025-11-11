# Development Guide

## Overview

Local AI is a privacy-focused AI application providing local speech-to-text, task management, and future AI capabilities. All processing happens locally without internet connectivity.

## Technology Stack

### Core Technologies

- **Python**: 3.13+ (strict requirement)
- **Package Manager**: `uv` (modern Python package manager)
- **Build Backend**: `uv_build`
- **Project Structure**: Standard Python package with `src/` layout

### Key Dependencies

- **Speech Recognition**: `faster-whisper` (local Whisper implementation)
- **Audio Processing**: `pyaudio` for microphone input
- **Voice Activity Detection**: `webrtcvad`
- **ML Framework**: `torch` (PyTorch) for GPU acceleration
- **System Monitoring**: `psutil` for performance metrics

### Development Tools

- **Testing**: `pytest` with async support (`pytest-asyncio`)
- **Coverage**: `pytest-cov` for test coverage reporting
- **Type Checking**: `mypy` with strict mode enabled
- **Linting/Formatting**: `ruff` (replaces flake8, black, isort)
- **Security**: `bandit` for security linting
- **Documentation**: `docformatter` for docstring formatting
- **Dead Code Detection**: `vulture` for finding unused code

## Development Setup

### Installing uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing the Project

```bash
# Install project in development mode with all dev dependencies
uv pip install -e .[dev]
```

## Testing

### Running Tests

Tests can be run directly with `pytest` (no `python -m` prefix needed):

```bash
# Run all tests in parallel (recommended)
pytest -n 2

# Run with coverage (required for all development)
pytest -n 2 --cov=src --cov-fail-under=90

# Run specific test categories
pytest -n 2 -m unit            # Unit tests only
pytest -n 2 -m integration     # Integration tests only
pytest -m performance          # Performance benchmarks (no parallel)

# TDD workflow
pytest -x --cov=src                         # Stop on first failure
pytest --cov=src --cov-report=term-missing  # Show missing coverage lines
pytest -n 2 -x                              # Parallel with stop on first failure

# Recommended workflow
pytest -n 2 -m unit && pytest -n 2 -m integration
```

### Test Categories

- **Unit tests** (`-m unit`): Fast, isolated component tests
- **Integration tests** (`-m integration`): Component interaction tests
- **Performance tests** (`-m performance`): Benchmarking and optimization tests

### Coverage Requirements

- **New code**: 90% minimum line coverage
- **Critical paths**: 100% coverage for core functionality (audio processing, task management)
- **Dead code monitoring**: Regular analysis to remove unused code

## Code Quality

### Type Checking

```bash
mypy src tests
```

### Linting and Formatting

```bash
# Check for issues
ruff check src tests

# Auto-format code
ruff format src tests

# Combined check and format
ruff check src tests && ruff format src tests
```

### Security Scanning

```bash
bandit -r src
```

### Dead Code Detection

```bash
vulture src tests
```

### All Quality Checks

```bash
# Run before commits
mypy src tests && ruff check src tests && ruff format src tests && bandit -r src
```

## Quality Maintenance Schedule

- **Before each commit**: `ruff check` and `ruff format`
- **Daily development**: `mypy` type checking
- **Weekly**: `vulture` dead code analysis
- **Weekly**: Full quality suite including `bandit`
- **Before releases**: Complete validation with coverage reports

## Test-Driven Development (TDD)

### TDD is Mandatory

All new features MUST follow TDD methodology:

1. **Write tests first** that define expected behavior and edge cases
2. **Implement minimal code** to make tests pass (Red-Green-Refactor)
3. **Refactor and optimize** while maintaining test coverage
4. **Maintain 90% minimum coverage** for all new code
5. **Regular dead code analysis** to remove unused functionality

### TDD Workflow Example

```bash
# 1. Write failing test
# 2. Run test to see it fail
pytest -x tests/test_new_feature.py

# 3. Implement minimal code
# 4. Run test to see it pass
pytest -x tests/test_new_feature.py

# 5. Check coverage
pytest --cov=src --cov-report=term-missing tests/test_new_feature.py

# 6. Refactor if needed
# 7. Run all tests to ensure nothing broke
pytest -n 2
```

## Running the Application

### After Installation

```bash
# Basic usage
local-ai

# With options
local-ai --verbose                      # Enable debug logging
local-ai --reset-model-cache           # Clear model cache
local-ai --reset-optimization-cache    # Clear optimization cache
```

### Module Approach

```bash
python -m local_ai.main
```

## Coding Standards

### Naming Conventions

- **Files & Modules**: `snake_case.py` (e.g., `audio_capture.py`)
- **Classes**: `PascalCase` (e.g., `SpeechToTextService`, `WhisperTranscriber`)
- **Exceptions**: End with `Error` (e.g., `AudioCaptureError`, `TranscriptionError`)
- **Functions & Methods**: `snake_case()` (e.g., `start_listening()`, `process_audio_chunk()`)
- **Async Methods**: Clearly marked with `async def`
- **Private Methods**: Prefix with underscore (e.g., `_process_audio_chunk()`)
- **Constants**: `UPPER_SNAKE_CASE` in `config.py` (e.g., `DEFAULT_SAMPLE_RATE`)

### Configuration Standards

- **Line Length**: 90 characters (ruff configuration)
- **Target Python**: 3.13
- **Import Sorting**: Handled by ruff
- **Type Hints**: Strict typing required (mypy strict mode)
- **Docstrings**: Google style docstrings required

### Architecture Patterns

#### Dependency Injection

Services accept dependencies in constructors to enable testing with mocks:

```python
class SpeechToTextService:
    def __init__(
        self,
        audio_capture: AudioCapture,
        transcriber: WhisperTranscriber,
        vad: VoiceActivityDetector
    ):
        self.audio_capture = audio_capture
        self.transcriber = transcriber
        self.vad = vad
```

#### Abstract Interfaces

Use ABC (Abstract Base Classes) for extensibility:

```python
from abc import ABC, abstractmethod

class ProcessingHandler(ABC):
    @abstractmethod
    async def process(self, data: Any) -> Any:
        pass
```

#### Async/Await

All I/O operations are async:

```python
async def process_audio(self) -> None:
    async for chunk in self.audio_capture.stream():
        result = await self.transcriber.transcribe(chunk)
```

#### Error Handling

- Custom exception hierarchy in `exceptions.py`
- Graceful degradation (GPU → CPU fallback)
- Comprehensive error logging

#### Configuration Management

- Centralized constants in `config.py`
- Runtime optimization based on system capabilities
- Caching of optimization decisions

## Greenfield Project Philosophy

This is a greenfield project in active development. **Breaking changes are acceptable.**

### Do Not Worry About Backwards Compatibility

When making improvements, feel free to change:

- API interfaces and method signatures
- Data models and class structures
- Configuration formats and constants
- CLI arguments and options
- File formats and storage schemas

### Prioritize

- Clean, well-designed interfaces
- Optimal performance and user experience
- Code maintainability and clarity
- Modern best practices

Focus on building the right solution rather than maintaining compatibility with earlier iterations.

## Module Organization

### Core Components

- **service.py**: Main orchestrator that coordinates all components
- **audio_capture.py**: Handles microphone input and audio streaming
- **vad.py**: Voice activity detection using WebRTC VAD
- **transcriber.py**: Whisper-based speech-to-text conversion
- **pipeline.py**: Extensible processing pipeline for future AI features

### Supporting Components

- **models.py**: Data classes (`AudioChunk`, `TranscriptionResult`, `SpeechSegment`)
- **config.py**: Configuration constants and settings
- **exceptions.py**: Custom exception hierarchy
- **optimization.py**: Performance optimization based on system capabilities
- **performance_monitor.py**: Metrics collection and monitoring
- **cache_utils.py**: Cache management and utilities
- **logging_utils.py**: Logging configuration and utilities

### Extensibility Layer

The `interfaces.py` module defines abstract base classes for extensible AI pipeline:

- `ProcessingHandler`: Base for all processing stages
- `EmbeddingHandler`: Text embedding generation
- `ResponseGenerationHandler`: AI response generation
- `TextToSpeechHandler`: Speech synthesis
- `ProcessingPipeline`: Pipeline management interface

## Test Organization

### Structure

- Mirror source structure in `tests/` directory
- One test file per source module (e.g., `test_audio_capture.py`)
- Test data in `tests/test_data/` with organized subdirectories

### Example

```
src/local_ai/speech_to_text/audio_capture.py
tests/test_speech_to_text/test_audio_capture.py
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've installed with `uv pip install -e .[dev]`
2. **Test failures**: Run with `-v` for verbose output: `pytest -v`
3. **Coverage too low**: Use `--cov-report=term-missing` to see uncovered lines
4. **Type errors**: Run `mypy src tests` to see all type issues

### Getting Help

- Check existing documentation in `docs/`
- Review test files for usage examples
- Check the steering rules in `.kiro/steering/`
