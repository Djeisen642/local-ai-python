# Design Document

## Overview

This design fixes two critical bugs in the audio processing pipeline that cause debug audio to sound cut off or sped up:

1. **Sample Rate Bug**: The `_create_wav_data` method incorrectly assumes the source sample rate equals the target sample rate, passing the same value for both parameters. This causes audio to be played at the wrong speed.

2. **VAD Frame Boundary Bug**: The `_process_audio_pipeline` method discards incomplete VAD frames at chunk boundaries, causing audio to be cut off.

The fixes are minimal, targeted, and add no processing overhead to the pipeline.

## Architecture

### Current Flow (Buggy)

```
AudioCapture (16kHz) → Service → Transcriber
                                    ↓
                              _convert_audio_format
                                    ↓
                              _create_wav_data(audio, 16000, 1, 16000)
                                    ↓
                              Assumes source=16000, but audio might be different!
```

### Fixed Flow

```
AudioCapture (16kHz) → Service (passes sample_rate) → Transcriber
                                                          ↓
                                                    _convert_audio_format(audio, 16000, actual_rate)
                                                          ↓
                                                    _create_wav_data(audio, actual_rate, 1, 16000)
```

## Components and Interfaces

### 1. Configuration (config.py)

Add two new constants:

- `VAD_PAD_INCOMPLETE_FRAMES = True` - Pad incomplete VAD frames with zeros
- `AUDIO_DEBUG_LOG_SAMPLE_RATES = False` - Log sample rates at each processing step

### 2. Transcriber (transcriber.py)

**Modified Methods:**

- `_convert_audio_format()` - Add optional `source_sample_rate` parameter, pass to `_create_wav_data`
- `_create_wav_data()` - Use actual `source_sample_rate` instead of assuming it equals `target_sample_rate`, skip resampling when rates match
- `transcribe_audio_with_result()` - Add optional `source_sample_rate` parameter, pass to `_convert_audio_format`

### 3. Service (service.py)

**Modified Methods:**

- `_process_audio_pipeline()` - Pad incomplete VAD frames when `VAD_PAD_INCOMPLETE_FRAMES` is True, log padding operations
- `_process_speech_segment()` - Pass sample rate to `transcribe_audio_with_result()`

### 4. AudioCapture (audio_capture.py)

No changes required - already stores correct sample rate.

### 5. AudioDebugger (audio_debugger.py)

Optional: Add sample rate to debug logs when `AUDIO_DEBUG_LOG_SAMPLE_RATES` is enabled.

## Data Models

No new data models required.

## Error Handling

- **Sample Rate Mismatch**: Log warning, proceed with resampling, fallback to minimal WAV on failure
- **Incomplete Frame Padding**: Pad with zeros if enabled, skip if disabled, never raise exceptions
- **Resampling Failure**: Log error, return original samples, continue processing

## Testing Strategy

### Unit Tests

1. Test correct sample rate passing through `_convert_audio_format` and `_create_wav_data`
2. Test resampling is skipped when source == target
3. Test resampling occurs when source != target
4. Test VAD frame padding when enabled/disabled
5. Test backward compatibility with default parameters

### Integration Tests

6. Test end-to-end audio quality (duration and speed)
7. Test with different sample rates (8kHz, 16kHz, 32kHz, 48kHz)

### Manual Testing

8. Verify debug audio sounds natural (not sped up or cut off)
9. Test configuration flags work as expected

## Performance Considerations

**Zero performance impact**:

- Sample rate fix only changes which value is passed
- Frame padding adds at most 960 bytes per incomplete frame (rare)
- Debug logging only active when enabled
- Resampling skip actually improves performance

**Memory impact**: Negligible (< 1KB per incomplete frame)

**Latency impact**: None (no new processing steps)

## Migration and Deployment

**100% backward compatible**:

- All new parameters have default values
- Existing code works without changes
- Default configuration maintains current behavior
- No migration needed
- Deploy without downtime
