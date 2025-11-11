# Project Structure

## Layout

```
src/local_ai/
├── main.py                          # CLI entry
├── speech_to_text/                  # STT module
│   ├── service.py                   # Main orchestrator
│   ├── audio_capture.py             # Mic input
│   ├── vad.py                       # Voice activity detection
│   ├── transcriber.py               # Whisper STT
│   ├── pipeline.py                  # Extensible processing
│   ├── models.py                    # Data classes
│   ├── config.py                    # Constants
│   ├── exceptions.py                # Custom errors
│   ├── optimization.py              # Performance tuning
│   ├── performance_monitor.py       # Metrics
│   ├── interfaces.py                # ABC for extensibility
│   └── audio_filtering/             # Audio enhancement
│       ├── audio_filter_pipeline.py # Filter orchestration
│       ├── noise_reduction.py       # Noise removal
│       ├── spectral_enhancer.py     # Spectral processing
│       ├── audio_normalizer.py      # Volume normalization
│       ├── adaptive_processor.py    # Adaptive filtering
│       ├── models.py                # Filter data classes
│       └── interfaces.py            # Filter ABCs
└── task_management/                 # Task/calendar module
    ├── database.py                  # SQLite persistence
    ├── models.py                    # Task data classes
    ├── config.py                    # Task constants
    ├── exceptions.py                # Task errors
    └── interfaces.py                # Task ABCs

tests/
├── test_speech_to_text/             # STT tests
├── test_task_management/            # Task tests
└── test_data/audio/                 # Test WAV files
```

## Naming

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`, exceptions end with `Error`
- **Functions**: `snake_case()`, async clearly marked, private with `_prefix()`
- **Constants**: `UPPER_SNAKE_CASE`

## Patterns

- **Dependency Injection**: Pass deps in constructors for testability
- **ABC Interfaces**: Extensible contracts for future features
- **Async/Await**: All I/O is async with proper context management
- **Error Handling**: Custom hierarchy, graceful degradation (GPU→CPU)
- **Config**: Centralized in `config.py`, runtime optimization with caching

## Testing (TDD Required)

- **Categories**: `-m unit` (fast), `-m integration` (interaction), `-m performance` (benchmarks)
- **Organization**: Mirror `src/` in `tests/`, one test file per module
- **Coverage**: 90% minimum for new code, 100% for critical paths
