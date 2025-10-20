# Project Structure

## Directory Layout

```
local-ai-python/
├── src/local_ai/                    # Main package source
│   ├── __init__.py
│   ├── main.py                      # CLI entry point
│   └── speech_to_text/              # Speech-to-text module
│       ├── __init__.py
│       ├── audio_capture.py         # Microphone input management
│       ├── cache_utils.py           # Cache management utilities
│       ├── cli_optimization.py      # CLI-specific optimizations
│       ├── config.py                # Configuration constants
│       ├── exceptions.py            # Custom exception classes
│       ├── interfaces.py            # Abstract interfaces & data models
│       ├── logging_utils.py         # Logging configuration
│       ├── models.py                # Data classes (AudioChunk, etc.)
│       ├── optimization.py          # Performance optimization
│       ├── optimization_cache.py    # Caching for optimizations
│       ├── performance_monitor.py   # Performance tracking
│       ├── pipeline.py              # Processing pipeline
│       ├── service.py               # Main orchestrator service
│       ├── transcriber.py           # Whisper transcription
│       └── vad.py                   # Voice activity detection
├── tests/                           # Test suite
│   ├── test_data/                   # Test audio files
│   │   └── audio/                   # WAV files for testing
│   └── test_speech_to_text/         # Module-specific tests
├── docs/                            # Documentation
├── .kiro/steering/                  # AI assistant steering rules
├── pyproject.toml                   # Project configuration
└── README.md                        # Project documentation
```

## Module Organization

### Core Components

- **service.py** - Main orchestrator that coordinates all components
- **audio_capture.py** - Handles microphone input and audio streaming
- **vad.py** - Voice activity detection using WebRTC VAD
- **transcriber.py** - Whisper-based speech-to-text conversion
- **pipeline.py** - Extensible processing pipeline for future AI features

### Supporting Components

- **models.py** - Data classes (`AudioChunk`, `TranscriptionResult`, `SpeechSegment`)
- **config.py** - Configuration constants and settings
- **exceptions.py** - Custom exception hierarchy
- **optimization.py** - Performance optimization based on system capabilities
- **performance_monitor.py** - Metrics collection and monitoring
- **cache_utils.py** - Cache management and utilities
- **logging_utils.py** - Logging configuration and utilities

### Extensibility Layer

- **interfaces.py** - Abstract base classes for extensible AI pipeline
  - `ProcessingHandler` - Base for all processing stages
  - `EmbeddingHandler` - Text embedding generation
  - `ResponseGenerationHandler` - AI response generation
  - `TextToSpeechHandler` - Speech synthesis
  - `ProcessingPipeline` - Pipeline management interface

## Naming Conventions

### Files & Modules

- Snake case: `audio_capture.py`, `voice_activity_detection.py`
- Descriptive names that indicate functionality
- Group related functionality in single modules

### Classes

- PascalCase: `SpeechToTextService`, `WhisperTranscriber`
- Descriptive names ending with purpose: `AudioCapture`, `VoiceActivityDetector`
- Exception classes end with `Error`: `AudioCaptureError`, `TranscriptionError`

### Functions & Methods

- Snake case: `start_listening()`, `process_audio_chunk()`
- Async methods clearly indicated: `async def process_audio()`
- Private methods prefixed with underscore: `_process_audio_chunk()`

### Constants

- UPPER_SNAKE_CASE in `config.py`: `DEFAULT_SAMPLE_RATE`, `VAD_FRAME_DURATION_MS`

## Architecture Patterns

### Dependency Injection

- Services accept dependencies in constructors
- Enables testing with mocks and different implementations

### Abstract Interfaces

- Use ABC (Abstract Base Classes) for extensibility
- Define clear contracts for future implementations

### Async/Await

- All I/O operations are async
- Use `asyncio` for concurrent processing
- Proper async context management

### Error Handling

- Custom exception hierarchy in `exceptions.py`
- Graceful degradation (GPU → CPU fallback)
- Comprehensive error logging

### Configuration Management

- Centralized constants in `config.py`
- Runtime optimization based on system capabilities
- Caching of optimization decisions

## Testing Structure

### Test-Driven Development (TDD)

**Required for all new features**: Write tests first, then implement (Red-Green-Refactor cycle)

### Test Categories

- **Unit tests** (`-m unit`) - Fast, isolated component tests
- **Integration tests** (`-m integration`) - Component interaction tests
- **Performance tests** (`-m performance`) - Benchmarking and optimization

### Test Organization

- Mirror source structure in `tests/` directory
- One test file per source module: `test_audio_capture.py`
- Test data in `tests/test_data/` with organized subdirectories

### Coverage Requirements

- **New code**: 90% minimum line coverage
- **Critical paths**: 100% coverage for core audio processing
- **Dead code monitoring**: Regular analysis to remove unused code
