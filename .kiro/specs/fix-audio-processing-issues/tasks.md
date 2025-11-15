# Implementation Plan

- [x] 1. Add configuration constants to config.py

  - Add `VAD_PAD_INCOMPLETE_FRAMES = True` constant
  - Add `AUDIO_DEBUG_LOG_SAMPLE_RATES = False` constant
  - Add inline documentation explaining when to use each flag
  - _Requirements: 4.1, 4.2, 4.4_

- [x] 2. Write unit tests for sample rate fixes (TDD - Red phase)

  - [x] 2.1 Test `_convert_audio_format` with source_sample_rate parameter

    - Test with explicit source rate
    - Test with default (None) source rate
    - Verify correct rate is passed to `_create_wav_data`
    - _Requirements: 1.1, 1.2, 5.1_

  - [x] 2.2 Test `_create_wav_data` resampling logic

    - Test resampling is skipped when source == target
    - Test resampling occurs when source != target
    - Test with various sample rates (8kHz, 16kHz, 32kHz, 48kHz)
    - Verify output duration matches input duration
    - _Requirements: 1.3, 1.4_

  - [x] 2.3 Test backward compatibility
    - Test calling methods without new parameters
    - Verify default behavior is preserved
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 3. Fix sample rate handling in transcriber.py (TDD - Green phase)

  - [x] 3.1 Update `_convert_audio_format` method signature

    - Add optional `source_sample_rate` parameter with default `None`
    - Default to `DEFAULT_SAMPLE_RATE` when not provided
    - Pass `source_sample_rate` to `_create_wav_data` call
    - _Requirements: 1.1, 1.2_

  - [x] 3.2 Fix `_create_wav_data` method

    - Use actual `source_sample_rate` parameter instead of `target_sample_rate`
    - Add check to skip resampling when `source_sample_rate == target_sample_rate`
    - Add debug logging when `AUDIO_DEBUG_LOG_SAMPLE_RATES` is True
    - Log resampling operations with source and target rates
    - _Requirements: 1.3, 1.4, 1.5_

  - [x] 3.3 Update `transcribe_audio_with_result` method signature
    - Add optional `source_sample_rate` parameter with default `None`
    - Pass `source_sample_rate` to `_convert_audio_format` call
    - _Requirements: 1.1, 3.2_

- [x] 4. Write unit tests for VAD frame padding (TDD - Red phase)

  - [x] 4.1 Test frame padding when enabled

    - Create audio chunk with incomplete frame
    - Verify frame is padded with zeros
    - Verify padded frame is processed by VAD
    - _Requirements: 2.1, 2.2_

  - [x] 4.2 Test frame skipping when disabled

    - Set `VAD_PAD_INCOMPLETE_FRAMES` to False
    - Create audio chunk with incomplete frame
    - Verify incomplete frame is skipped
    - _Requirements: 2.5_

  - [x] 4.3 Test sample rate propagation
    - Verify Service passes sample rate to Transcriber
    - Verify sample rate is consistent throughout pipeline
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 5. Fix VAD frame handling in service.py (TDD - Green phase)

  - [x] 5.1 Update `_process_audio_pipeline` method

    - Add frame padding logic when `VAD_PAD_INCOMPLETE_FRAMES` is True
    - Pad incomplete frames with zeros to complete them
    - Add trace logging when padding occurs
    - Skip incomplete frames when `VAD_PAD_INCOMPLETE_FRAMES` is False
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 5.2 Update `_process_speech_segment` method
    - Get sample rate from pipeline configuration
    - Pass `source_sample_rate` parameter to `transcribe_audio_with_result` call
    - _Requirements: 3.2, 3.3_

- [x] 6. Add optional debug logging to audio_debugger.py

  - Update `save_audio_sync` method to log sample rate when `AUDIO_DEBUG_LOG_SAMPLE_RATES` is True
  - Include sample rate and duration in debug log message
  - _Requirements: 3.4, 4.3_

- [ ] 7. Write integration tests

  - [ ] 7.1 Test end-to-end audio quality

    - Capture audio, process through pipeline, save debug file
    - Verify debug audio duration matches input duration (within 1% tolerance)
    - Verify no audio quality degradation
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

  - [ ] 7.2 Test with different sample rates
    - Test with 8kHz, 16kHz, 32kHz, 48kHz
    - Verify all rates are handled correctly
    - Verify resampling occurs only when needed
    - _Requirements: 1.3, 1.4, 3.1, 3.2_

- [ ] 8. Manual verification
  - Enable audio debugging
  - Speak into microphone
  - Listen to saved WAV files
  - Verify audio sounds natural (not sped up or cut off)
  - Test with `VAD_PAD_INCOMPLETE_FRAMES` enabled and disabled
  - Test with `AUDIO_DEBUG_LOG_SAMPLE_RATES` enabled and disabled
  - _Requirements: All_
