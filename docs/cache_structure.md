# Cache Structure

The Local AI speech-to-text system uses a unified cache structure under `~/.cache/local_ai/speech_to_text/`.

## Directory Structure

```
~/.cache/local_ai/speech_to_text/
├── optimization/                    # Optimization cache
│   ├── system_capabilities.json    # System hardware/software capabilities
│   ├── optimized_configs.json      # Generated optimization configurations
│   └── performance_history.json    # Performance metrics history
└── models/                          # AI models cache
    └── whisper/                     # Whisper speech-to-text models
        ├── config.json              # Model configuration
        ├── vocabulary.txt           # Model vocabulary
        ├── tokenizer.json           # Tokenizer configuration
        └── model.bin                # Model weights (large file)
```

## Cache Management

### Via Main CLI

- `python -m local_ai.main --reset-model-cache` - Clear Whisper models cache
- `python -m local_ai.main --reset-optimization-cache` - Clear optimization cache

### Via Optimization CLI

- `python -m local_ai.speech_to_text.cli_optimization info` - Show cache information
- `python -m local_ai.speech_to_text.cli_optimization clear-cache --type all` - Clear all caches
- `python -m local_ai.speech_to_text.cli_optimization clear-cache --type models` - Clear models only
- `python -m local_ai.speech_to_text.cli_optimization clear-cache --type system` - Clear system capabilities
- `python -m local_ai.speech_to_text.cli_optimization clear-cache --type config` - Clear optimization configs
- `python -m local_ai.speech_to_text.cli_optimization clear-cache --type performance` - Clear performance history

## Benefits

1. **Unified Location**: All caches are organized under a single directory tree
2. **Separation of Concerns**: Models and optimization data are in separate subdirectories
3. **Easy Management**: Clear individual cache types or everything at once
4. **Visibility**: Easy to see cache sizes and contents with the info command
5. **Backward Compatibility**: Still clears legacy HuggingFace cache locations when needed

## Cache Sizes

- **Optimization cache**: Usually < 1 KB (JSON files with system info and configs)
- **Whisper models cache**: ~500-900 MB per model (depending on model size)
  - small model: ~244 MB
  - medium model: ~769 MB
  - large model: ~1550 MB

## Cache Validity

- **System capabilities**: 24 hours (hardware doesn't change often)
- **Optimization configs**: 1 hour (allows for quick iteration)
- **Performance history**: 1 week (longer-term performance trends)
- **Models**: Permanent until manually cleared (models don't change)
