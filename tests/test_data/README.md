# Test Audio Files

This directory contains audio files for testing the WhisperTranscriber functionality.

## File Structure

```
tests/test_data/audio/
├── hello_world.wav          # "Hello world" - basic functionality test
├── numbers.wav              # "One two three four five" - number recognition
├── alphabet.wav             # "A B C D E F G" - letter recognition
├── short_sentence.wav       # "This is a test sentence" - longer phrase
├── edge_cases/
│   ├── silence.wav          # 2 seconds of silence
│   └── very_short.wav       # Very short audio (~1 second)
├── quality/
│   ├── high_quality_16khz.wav  # High quality 16kHz audio
│   └── low_quality_8khz.wav    # Lower quality 8kHz audio
└── scenarios/
    ├── command.wav          # "Start recording now" - command recognition
    ├── dictation.wav        # "Please transcribe this message" - dictation
    └── question.wav         # "What time is it" - question recognition
```

## Audio Specifications

- **Format**: WAV (RIFF/WAVE)
- **Encoding**: 16-bit PCM
- **Channels**:
  - Most files: Stereo (2 channels) at 44.1kHz
  - Quality test files: Mono (1 channel)
- **Sample Rates**:
  - Standard files: 44.1kHz stereo (music/recording quality)
  - High quality: 16kHz mono (Whisper's optimal format)
  - Low quality: 8kHz mono (phone/low-bandwidth quality)

## Test Results

When tested with WhisperTranscriber, these files produce the following transcriptions:

| File                             | Expected Content                 | Actual Transcription              |
| -------------------------------- | -------------------------------- | --------------------------------- |
| `hello_world.wav`                | "Hello world"                    | "Hello world."                    |
| `numbers.wav`                    | "One two three four five"        | "One, two, three, four, five."    |
| `alphabet.wav`                   | "A B C D E F G"                  | "A, B, C, D, E, F, G."            |
| `short_sentence.wav`             | "This is a test sentence"        | "This is a test sentence."        |
| `edge_cases/silence.wav`         | (silence)                        | "" (empty)                        |
| `edge_cases/very_short.wav`      | "Hi"                             | "high."                           |
| `quality/high_quality_16khz.wav` | "Hello world"                    | "Hello world."                    |
| `quality/low_quality_8khz.wav`   | "Hello world"                    | "Hello world."                    |
| `scenarios/command.wav`          | "Start recording now"            | "Start recording now."            |
| `scenarios/dictation.wav`        | "Please transcribe this message" | "please transcribe this message." |
| `scenarios/question.wav`         | "What time is it"                | "what time is it"                 |

## Usage in Tests

These files are used in two types of tests:

### Unit Tests (`test_transcriber.py`)

- Fast, mocked tests that don't use real audio
- Test error handling, edge cases, and interface contracts
- Run quickly in CI/CD environments

### Integration Tests (`test_transcriber_integration.py`)

- Use real audio files and actual Whisper model
- Test actual transcription accuracy and performance
- Marked with `@pytest.mark.integration`
- May be slower and require model downloads

## Running Tests

```bash
# Run only fast unit tests
pytest tests/test_speech_to_text/test_transcriber.py

# Run only integration tests
pytest tests/test_speech_to_text/test_transcriber_integration.py -m integration

# Run all transcriber tests
pytest tests/test_speech_to_text/test_transcriber*.py

# Skip integration tests (for CI)
pytest tests/test_speech_to_text/ -m "not integration"
```

## Notes

- The MP3 test file is missing and will be skipped
- Integration tests require the faster-whisper model to be available
- Tests are designed to be flexible with transcription variations (punctuation, capitalization)
- Very short audio may not transcribe perfectly due to Whisper's limitations
