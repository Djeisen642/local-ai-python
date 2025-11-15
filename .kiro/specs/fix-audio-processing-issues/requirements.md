# Requirements Document

## Introduction

This feature addresses audio quality issues in the speech-to-text pipeline where debug audio files sometimes sound cut off or sped up. The root causes are:

1. **Incorrect resampling** in the transcriber's `_create_wav_data` method - it assumes the source sample rate equals the target sample rate, causing audio to be played at the wrong speed
2. **VAD frame boundary issues** where incomplete audio frames at chunk boundaries are discarded, causing audio to be cut off
3. **Lack of validation** to detect when audio is being processed incorrectly

The solution fixes these bugs and adds minimal configuration flags for debugging, without adding processing overhead to the pipeline.

## Glossary

- **System**: The speech-to-text service and its components (AudioCapture, VAD, Transcriber)
- **Resampling**: Converting audio from one sample rate to another
- **VAD**: Voice Activity Detection - determines which audio contains speech
- **VAD Frame**: Fixed-size audio chunk required by WebRTC VAD (e.g., 30ms at 16kHz)
- **Sample Rate**: Number of audio samples per second (Hz)
- **Audio Chunk**: Raw audio data captured from microphone
- **Debug Audio**: WAV files saved for debugging transcription issues
- **Linear Interpolation**: Simple resampling method that can introduce artifacts
- **Frame Boundary**: The alignment of audio data with VAD's expected frame size

## Requirements

### Requirement 1: Fix Incorrect Sample Rate Assumption

**User Story:** As a user, I want audio to be transcribed at the correct speed, so that debug audio files sound natural and accurate.

#### Acceptance Criteria

1. WHEN the Transcriber calls `_convert_audio_format`, THE Transcriber SHALL pass the actual source sample rate from AudioCapture configuration
2. WHEN the Transcriber calls `_create_wav_data`, THE Transcriber SHALL use the correct source sample rate instead of assuming it equals the target rate
3. WHEN audio is already at the target sample rate, THE Transcriber SHALL skip resampling entirely
4. WHEN the source and target sample rates differ, THE Transcriber SHALL resample using the correct ratio
5. THE Transcriber SHALL log at debug level when resampling occurs with source and target rates

### Requirement 2: Fix VAD Frame Boundary Handling

**User Story:** As a user, I want all captured audio to be processed, so that speech is not cut off at chunk boundaries.

#### Acceptance Criteria

1. WHEN the Service processes an audio chunk AND the chunk contains incomplete VAD frames, THE Service SHALL pad the incomplete frame with zeros to make it complete
2. WHEN padding is applied, THE Service SHALL process the padded frame through VAD
3. WHEN the Service combines speech chunks for transcription, THE Service SHALL include all audio data without dropping incomplete frames
4. THE Service SHALL log at trace level when frame padding occurs
5. THE System SHALL add a configuration flag `VAD_PAD_INCOMPLETE_FRAMES` defaulting to True

### Requirement 3: Sample Rate Consistency

**User Story:** As a developer, I want the system to use consistent sample rates, so that audio is not inadvertently resampled incorrectly.

#### Acceptance Criteria

1. WHEN AudioCapture initializes, THE AudioCapture SHALL store the configured sample rate
2. WHEN the Transcriber converts audio format, THE Transcriber SHALL receive the source sample rate from AudioCapture
3. WHEN the Service passes audio to the Transcriber, THE Service SHALL include sample rate metadata
4. THE System SHALL ensure the same sample rate (16kHz) is used throughout the pipeline by default
5. THE System SHALL log a warning if sample rate changes are detected between components

### Requirement 4: Minimal Debug Configuration

**User Story:** As a developer debugging audio issues, I want minimal configuration options to test fixes, so that I can verify the bugs are resolved without adding complexity.

#### Acceptance Criteria

1. THE System SHALL add a configuration flag `VAD_PAD_INCOMPLETE_FRAMES` to enable/disable frame padding (default: True)
2. THE System SHALL add a configuration flag `AUDIO_DEBUG_LOG_SAMPLE_RATES` to enable sample rate logging (default: False)
3. WHEN `AUDIO_DEBUG_LOG_SAMPLE_RATES` is True, THE System SHALL log sample rates at each processing step
4. THE System SHALL document these flags in config.py with clear explanations
5. THE System SHALL not add any other configuration flags or processing options

### Requirement 5: Preserve Existing Behavior

**User Story:** As an existing user, I want the fixes to improve audio quality without changing the pipeline, so that performance is not impacted.

#### Acceptance Criteria

1. THE System SHALL not add new processing steps to the audio pipeline
2. THE System SHALL not increase processing latency
3. THE System SHALL not change the API signatures of any public methods
4. THE System SHALL maintain the same audio file formats and naming conventions
5. THE System SHALL only fix the identified bugs without refactoring unrelated code
