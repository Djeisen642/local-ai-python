# Implementation Plan (Test-Driven Development)

- [x] 1. TDD: AudioDebugger class core functionality
- [x] 1.1 Write tests for AudioDebugger initialization and basic methods

  - Write failing tests for `AudioDebugger` initialization with enabled/disabled states
  - Write failing tests for `is_enabled()` method
  - Write failing tests for directory creation on initialization
  - _Requirements: 2.1, 4.1_

- [x] 1.2 Implement AudioDebugger class to pass initialization tests

  - Create `src/local_ai/speech_to_text/audio_debugger.py` file
  - Implement `AudioDebugger.__init__` with `enabled` and `output_dir` parameters
  - Implement `is_enabled()` method
  - Implement automatic directory creation with error handling
  - _Requirements: 2.1, 4.1, 4.2_

- [x] 1.3 Write tests for audio file saving functionality

  - Write failing tests for `save_audio_sync()` method with valid audio data
  - Write failing tests for timestamped filename generation
  - Write failing tests for WAV file format and content validation
  - Write failing tests for duration calculation in filename
  - _Requirements: 1.1, 1.3, 3.1, 3.2, 3.3_

- [x] 1.4 Implement audio saving functionality to pass tests

  - Implement `save_audio_sync()` method using Python's `wave` module
  - Implement timestamped filename generation with format `audio_{date}_{time}_{duration_ms}.wav`
  - Implement duration calculation from audio data length and sample rate
  - Ensure proper WAV file headers (16kHz, mono, 16-bit)
  - _Requirements: 1.1, 1.3, 3.1, 3.2, 3.3_

- [x] 1.5 Write tests for error handling

  - Write tests for handling non-writable directories
  - Write tests for handling disk full scenarios
  - Write tests for handling invalid audio data
  - Verify that errors are logged but don't raise exceptions
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.3_

- [x] 1.6 Implement error handling to pass tests

  - Add try/except blocks for file system operations
  - Add logging for errors without raising exceptions
  - Handle edge cases (empty audio, invalid paths, etc.)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.3_

- [x] 2. Add configuration constants

  - Add audio debugging constants to `src/local_ai/speech_to_text/config.py`
  - Define `AUDIO_DEBUG_ENABLED`, `AUDIO_DEBUG_DEFAULT_DIR`, and `AUDIO_DEBUG_FILENAME_FORMAT`
  - _Requirements: 2.1, 4.3_

- [x] 3. TDD: WhisperTranscriber integration
- [x] 3.1 Write tests for WhisperTranscriber with AudioDebugger

  - Write failing tests for `WhisperTranscriber` accepting optional `audio_debugger` parameter
  - Write failing tests verifying audio is saved when debugger is enabled
  - Write failing tests verifying audio is not saved when debugger is disabled
  - Write failing tests verifying transcription continues when audio saving fails
  - _Requirements: 1.1, 1.4, 5.3_

- [x] 3.2 Implement WhisperTranscriber integration to pass tests

  - Modify `WhisperTranscriber.__init__` to accept optional `audio_debugger` parameter
  - Add audio debugging call in `transcribe_audio_with_result` after audio conversion
  - Wrap debugging call in try/except to ensure errors don't interrupt transcription
  - Add debug logging for audio save operations
  - _Requirements: 1.1, 1.4, 5.3_

- [x] 4. TDD: CLI argument parsing
- [x] 4.1 Write tests for CLI argument parsing

  - Write failing tests for `--debug-audio` flag parsing
  - Write failing tests for `--debug-audio-dir` optional argument parsing
  - Write failing tests for default values when flags are not provided
  - _Requirements: 2.1, 2.2, 4.1, 4.2_

- [x] 4.2 Implement CLI argument parsing to pass tests

  - Add `--debug-audio` flag to CLI argument parser in `src/local_ai/main.py`
  - Add `--debug-audio-dir` optional argument for custom output directory
  - Set appropriate default values and help text
  - _Requirements: 2.1, 2.2, 4.1, 4.2_

- [x] 5. TDD: SpeechToTextService integration
- [x] 5.1 Write tests for SpeechToTextService with AudioDebugger

  - Write failing tests for service initialization with audio debugging enabled
  - Write failing tests verifying `AudioDebugger` is passed to `WhisperTranscriber`
  - Write failing tests for service initialization with audio debugging disabled
  - _Requirements: 2.1, 2.3_

- [x] 5.2 Implement SpeechToTextService integration to pass tests

  - Modify `SpeechToTextService.initialize` to create `AudioDebugger` instance if enabled
  - Pass `AudioDebugger` instance to `WhisperTranscriber` constructor
  - Handle configuration from CLI arguments
  - _Requirements: 2.1, 2.3_

- [x] 6. Integration testing and validation
- [x] 6.1 Write end-to-end integration tests

  - Write tests for complete flow from CLI args to file creation
  - Write tests verifying WAV file format matches Whisper input exactly
  - Write tests for multiple transcriptions creating multiple files
  - Write tests for custom output directory configuration
  - _Requirements: 1.1, 1.3, 2.1, 4.1_

- [x] 6.2 Run integration tests and verify functionality

  - Run all integration tests and ensure they pass
  - Verify WAV files can be played back and analyzed
  - Verify no performance degradation when debugging is disabled
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7. Documentation
  - Add audio debugging section to README.md
  - Document `--debug-audio` and `--debug-audio-dir` CLI flags
  - Add usage examples for enabling audio debugging
  - Add troubleshooting notes for common issues (permissions, disk space)
  - Document WAV file format and naming convention
  - _Requirements: 2.1, 4.1_
