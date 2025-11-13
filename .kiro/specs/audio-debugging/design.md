# Audio Debugging Design Document

## Overview

The audio debugging feature provides a simple way to save the processed audio that is sent to the faster-whisper transcription model as WAV files. This helps debug transcription issues and understand what audio Whisper actually receives.

**Key Design Decisions:**

- Minimal implementation - just save WAV files with timestamps
- Configuration via single CLI flag
- Integration at the transcriber level
- Zero overhead when disabled
- Errors don't interrupt transcription

## Architecture

The audio debugger hooks into the transcription pipeline right before audio is sent to Whisper:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Audio Buffer   │───▶│  Transcription  │
│   (Microphone)  │    │   & VAD         │    │ (faster-whisper)│
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Audio Debugger  │
                                              │  (if enabled)   │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  WAV Files      │
                                              │  (timestamped)  │
                                              └─────────────────┘
```

## Components and Interfaces

### 1. AudioDebugger Class

**Purpose**: Simple audio capture to WAV files

```python
class AudioDebugger:
    def __init__(
        self,
        enabled: bool = False,
        output_dir: Path | None = None
    )
    def save_audio_sync(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Path | None
    def is_enabled(self) -> bool
```

**Key Features:**

- Simple synchronous file writing
- Automatic directory creation
- Timestamped filenames
- Error handling that doesn't interrupt transcription

### 2. Configuration Integration

**Purpose**: Simple CLI flag to enable audio debugging

**CLI Arguments:**

```python
--debug-audio                    # Enable audio debugging
--debug-audio-dir PATH          # Optional: custom output directory
```

### 3. WhisperTranscriber Integration

**Purpose**: Hook audio debugging into the transcription pipeline

**Modified Method:**

```python
async def transcribe_audio_with_result(
    self, audio_data: bytes
) -> TranscriptionResult:
    # ... existing validation ...

    # Convert audio to Whisper-compatible format
    converted_audio = self._convert_audio_format(
        audio_data, target_sample_rate=DEFAULT_SAMPLE_RATE
    )

    # Debug: Save audio if debugging is enabled (simple, non-blocking)
    if self._audio_debugger and self._audio_debugger.is_enabled():
        try:
            self._audio_debugger.save_audio_sync(
                converted_audio,
                sample_rate=DEFAULT_SAMPLE_RATE
            )
        except Exception as e:
            logger.debug(f"Audio debug save failed: {e}")

    # ... continue with transcription ...
```

## Data Models

No additional data models needed - uses existing Path and basic types.

## File Naming and Organization

### Directory Structure

```
~/.cache/local_ai/audio_debug/
├── audio_20241112_143022_500ms.wav
├── audio_20241112_143025_1200ms.wav
├── audio_20241112_143028_800ms.wav
└── ...
```

### Filename Format

```
audio_{YYYYMMDD}_{HHMMSS}_{duration_ms}.wav
```

**Example:** `audio_20241112_143022_500.wav`

- Date: November 12, 2024
- Time: 14:30:22
- Duration: 500ms

### WAV File Format

- **Sample Rate**: 16000 Hz (matches Whisper input)
- **Channels**: 1 (mono)
- **Sample Width**: 16-bit
- **Format**: Standard WAV with proper headers

## Error Handling

### File System Errors

- **Directory not writable**: Log error, continue transcription
- **Disk full**: Log error, continue transcription
- **Permission denied**: Log error, continue transcription

All errors are caught and logged but don't interrupt transcription.

## Testing Strategy

### Unit Tests

- **AudioDebugger**: Test file creation, naming, directory creation
- **Configuration**: Test CLI argument parsing
- **Integration**: Test WhisperTranscriber integration with mocked debugger

### Integration Tests

- **End-to-end**: Test complete flow from audio capture to file creation
- **Error handling**: Test graceful handling of file system errors

## Performance Optimization

### Minimal Overhead When Disabled

When audio debugging is disabled, the overhead is a single boolean check:

```python
if self._audio_debugger and self._audio_debugger.is_enabled():
    # Only executed when enabled
    self._audio_debugger.save_audio_sync(...)
```

### Simple Synchronous Writes

File writes are synchronous but fast (typically <10ms for small WAV files). The try/except ensures transcription continues even if writes fail.

## Configuration Constants

```python
# Audio Debugging Configuration
AUDIO_DEBUG_ENABLED = False  # Default disabled
AUDIO_DEBUG_DEFAULT_DIR = Path.home() / ".cache" / "local_ai" / "audio_debug"
AUDIO_DEBUG_FILENAME_FORMAT = "audio_{date}_{time}_{duration_ms}.wav"
```

## Integration with Existing Code

### Minimal Changes Required

1. **Add AudioDebugger class** to `src/local_ai/speech_to_text/audio_debugger.py`
2. **Update WhisperTranscriber** to accept optional AudioDebugger instance
3. **Update CLI argument parsing** in `src/local_ai/main.py`
4. **Update SpeechToTextService** to initialize AudioDebugger if enabled
5. **Add configuration constants** to `src/local_ai/speech_to_text/config.py`

### Backward Compatibility

- Feature is opt-in (disabled by default)
- No changes to existing APIs
- No impact on performance when disabled
- Graceful degradation on errors

## Usage Examples

### Enable via CLI

```bash
# Basic usage - saves to default directory
local-ai --debug-audio

# With custom directory
local-ai --debug-audio --debug-audio-dir /tmp/audio_debug
```

### Programmatic Usage

```python
from local_ai.speech_to_text.audio_debugger import AudioDebugger
from local_ai.speech_to_text.transcriber import WhisperTranscriber

# Create debugger
debugger = AudioDebugger(
    enabled=True,
    output_dir=Path("/tmp/audio_debug")
)

# Create transcriber with debugger
transcriber = WhisperTranscriber(
    model_size="small",
    audio_debugger=debugger
)

# Use normally - audio will be saved automatically
result = await transcriber.transcribe_audio_with_result(audio_data)
```

## Security Considerations

### Privacy

- Audio files contain raw speech data - users should be aware
- Default directory is in user's cache (not shared)
- Files are only readable by the user (0600 permissions)

### Disk Space

- No automatic cleanup - user manages files manually
- Files are small (typically <100KB each)
- User can delete directory contents anytime

### File System Access

- Only writes to configured directory
- Validates directory paths before writing
- Handles permission errors gracefully
