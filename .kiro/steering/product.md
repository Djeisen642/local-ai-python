# Product Overview

Local AI is a privacy-focused AI application that provides AI capabilities without requiring internet connectivity. The primary focus is on speech-to-text functionality using OpenAI's Whisper model running locally.

## Core Features

- **Real-time speech transcription** - Convert voice to text with local processing
- **Voice activity detection** - Automatically detects when speech is present
- **GPU acceleration** - Automatic GPU detection with CPU fallback
- **Privacy-first** - All processing happens locally, no data leaves the machine
- **Linux optimized** - Designed for Linux environments with headless support

## Planned Features

- Text-to-speech synthesis
- Email notification intelligence
- Calendar event prioritization
- Adaptive learning and personalization
- Smart home device integration

## Architecture Philosophy

The system is built with extensibility in mind, using abstract interfaces and plugin-style architecture to enable future AI assistant workflows including embedding generation, response generation, and text-to-speech capabilities.

## Development Philosophy

### Greenfield Project Approach

This is a greenfield project in active development. **Do not worry about backwards compatibility** when making improvements or changes to:

- API interfaces and method signatures
- Data models and class structures
- Configuration formats and constants
- CLI arguments and options
- File formats and storage schemas

**Prioritize:**

- Clean, well-designed interfaces
- Optimal performance and user experience
- Code maintainability and clarity
- Modern best practices

**Breaking changes are acceptable** as we iterate toward the best possible design. Focus on building the right solution rather than maintaining compatibility with earlier iterations.

### Test-Driven Development Mandate

**All new features MUST follow TDD methodology:**

1. **Write tests first** that define expected behavior and edge cases
2. **Implement minimal code** to make tests pass
3. **Refactor and optimize** while maintaining test coverage
4. **Maintain 90% minimum coverage** for all new code
5. **Regular dead code analysis** to remove unused functionality

**Coverage Standards:**

- **New modules**: 90% minimum line coverage
- **Core functionality**: 100% coverage for critical paths
- **Dead code removal**: Monthly analysis and cleanup

This ensures reliable, maintainable software while moving fast in a greenfield environment.
