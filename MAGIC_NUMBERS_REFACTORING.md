# Magic Numbers Refactoring Summary

## Overview

Extracted magic numbers from the `speech_to_text` module and converted them into named constants in `config.py`.

## Categories of Constants Added

### 1. Audio Capture Constants

- `AUDIO_LEVEL_LOG_INTERVAL` - Logging interval for audio levels
- `AUDIO_LEVEL_THRESHOLD` - Threshold for significant audio activity
- `AUDIO_SAMPLE_MAX_VALUE` - Maximum value for 16-bit audio
- `AUDIO_SAMPLE_NORMALIZATION` - Normalization factor for 16-bit audio

### 2. VAD (Voice Activity Detection) Constants

- `VAD_MIN_SAMPLES_FOR_ADAPTATION` - Minimum samples for threshold adaptation
- `VAD_ADAPTIVE_MULTIPLIER_*` - Multipliers for adaptive pause thresholds
- `VAD_DEBUG_LOG_INTERVAL` - Logging interval for VAD statistics

### 3. Transcriber Constants

- `TRANSCRIBER_CHANNELS_MONO` - Mono audio channel count
- `TRANSCRIBER_SAMPLE_WIDTH` - Sample width in bytes
- `TRANSCRIBER_SEGMENT_RATIO_*` - Signal ratio thresholds

### 4. Service Constants

- `SERVICE_TRANSCRIPTION_WINDOW` - Window for transcription stats
- `SERVICE_PERFORMANCE_WINDOW` - Window for performance stats
- `SERVICE_HIGH_LATENCY_WARNING` - Latency warning threshold

### 5. Optimization Constants (30+ constants)

- CPU core thresholds
- Processing intervals
- Buffer sizes
- Aggressiveness levels
- Adaptation thresholds

### 6. Performance Monitor Constants

- `PERF_TRANSCRIPTION_LATENCY_HIGH` - High latency threshold
- `PERF_SUCCESS_RATE_LOW` - Minimum acceptable success rate
- `PERF_STATS_WINDOW_*` - Statistics window durations

### 7. Audio Filtering Constants (100+ constants)

#### Noise Reduction (NR\_\*)

- FFT parameters
- Wiener filter coefficients
- SNR thresholds
- Spectral subtraction factors
- Noise type detection thresholds

#### Spectral Enhancement (SE\_\*)

- Filter orders and cutoffs
- Speech band frequencies
- Enhancement factors
- Echo reduction parameters
- Transient suppression thresholds

#### Normalization (NORM\_\*)

- Target levels and gains
- Compression ratios
- Attack/release times
- Limiter thresholds

#### Adaptive Processor (AP\_\*)

- FFT sizes and hop sizes
- Formant frequencies
- Speech detection thresholds
- Harmonic analysis parameters
- Learning rates
- SNR thresholds
- Scoring weights and penalties

#### Pipeline (AFP\_\*)

- Quality score thresholds
- Clipping detection
- Signal loss penalties

## Benefits

1. **Maintainability**: All magic numbers are now in one place with descriptive names
2. **Documentation**: Each constant has a comment explaining its purpose
3. **Consistency**: Related constants are grouped together
4. **Testability**: Constants can be easily mocked or overridden in tests
5. **Discoverability**: Developers can see all tunable parameters in one file

## Completed Work

### Phase 1: Constants Definition ✅

All ~180 constants have been defined in `config.py` with descriptive names and comments.

### Phase 2: Code Refactoring ✅

**optimization.py** has been fully refactored to use the new constants:

- Replaced all magic numbers in `_generate_optimized_config()`
- Replaced magic numbers in `optimize_for_latency()`
- Replaced magic numbers in `optimize_for_accuracy()`
- Replaced magic numbers in `optimize_for_resource_usage()`
- Replaced magic numbers in `AdaptiveOptimizer` class
- Added proper type annotation for `performance_history`
- Fixed return type issue in `should_adapt()` by wrapping in `bool()`

### Constants Added for optimization.py

- `OPT_DEFAULT_MEMORY_GB` - Default memory estimate
- `OPT_DEFAULT_GPU_MEMORY_GB` - Default GPU memory
- `OPT_MODEL_SIZE_*` - Model size constants (default, large, medium, small, tiny)
- `OPT_COMPUTE_TYPE_*` - Compute type constants (int8, float16)
- `OPT_DEVICE_*` - Device constants (cpu, cuda)
- `OPT_PLATFORM_*` - Platform constants (Linux)
- `OPT_PROCESSING_INTERVAL_*` - Processing interval constants
- `OPT_MAX_CONCURRENT_TRANSCRIPTIONS` - Max concurrent transcriptions
- `OPT_MAX_AUDIO_BUFFER_*` - Audio buffer size constants
- `OPT_CPU_CORES_*` - CPU core threshold constants
- `OPT_VAD_AGGRESSIVENESS_*` - VAD aggressiveness adjustment constants
- `OPT_ADAPTIVE_*` - Adaptive optimizer constants

## Next Steps

The remaining files still need refactoring:

1. `audio_capture.py` - Replace magic numbers with constants
2. `vad.py` - Replace magic numbers with constants
3. `transcriber.py` - Replace magic numbers with constants
4. `service.py` - Replace magic numbers with constants
5. `performance_monitor.py` - Replace magic numbers with constants
6. Audio filtering modules - Replace magic numbers with constants
7. Run tests to ensure functionality is preserved
8. Update any documentation that references these values

## Files Analyzed

- `audio_capture.py`
- `vad.py`
- `transcriber.py`
- `service.py`
- `optimization.py`
- `performance_monitor.py`
- `audio_filtering/audio_filter_pipeline.py`
- `audio_filtering/noise_reduction.py`
- `audio_filtering/spectral_enhancer.py`
- `audio_filtering/audio_normalizer.py`
- `audio_filtering/adaptive_processor.py`

## Total Constants Added

Approximately **150+ new constants** were added to `config.py`, covering all major subsystems of the speech-to-text module.
