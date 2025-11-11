# Audio Filtering Evaluation Results

## Test Date: November 10, 2025

## Executive Summary

Audio filtering was tested on three scripted audio files with varying complexity. **The results show that audio filtering provides minimal to no improvement in transcription accuracy while significantly increasing processing time.**

### Key Findings

- **Accuracy Impact**: -0.40% average (slightly worse with filtering)
- **Processing Time**: ~10x slower with filtering (4.11s → 37.90s for script1)
- **Recommendation**: **Disable audio filtering for clean audio sources**

## Detailed Results

### Script 1: Dates, Numbers, and Times

**Content**: Complex speech with dates (November 10th, 2025), number sequences (1-9, 0), large numbers (4,781,309), and times (9:10 PM EST).

| Metric          | Without Filtering | With Filtering | Change |
| --------------- | ----------------- | -------------- | ------ |
| WER             | 20.24%            | 20.24%         | 0.00%  |
| Processing Time | 4.11s             | 37.90s         | +822%  |

**Analysis**: Identical transcription quality, but filtering added 33.79 seconds of processing overhead.

### Script 2: Punctuation and Capitalization

**Content**: Tests proper nouns (Ms. Eleanor Vance, Aperture Technologies, Inc.), locations (San Francisco, California), and punctuation handling.

| Metric          | Without Filtering | With Filtering | Change |
| --------------- | ----------------- | -------------- | ------ |
| WER             | 0.00%             | 1.20%          | -1.20% |
| Processing Time | ~4s               | ~38s           | +850%  |

**Analysis**: Filtering actually **degraded** transcription quality slightly while adding massive processing overhead.

### Script 3: Homophones and Technical Vocabulary

**Content**: Tests homophones (their/there/they're, suite/sweet) and technical terms (JSON, OAuth 2.0, parse/perceive).

| Metric          | Without Filtering | With Filtering | Change |
| --------------- | ----------------- | -------------- | ------ |
| WER             | 5.49%             | 5.49%          | 0.00%  |
| Processing Time | ~4s               | ~38s           | +850%  |

**Analysis**: No improvement in handling homophones or technical vocabulary.

## Overall Summary

| Script      | WER (No Filter) | WER (Filtered) | Improvement |
| ----------- | --------------- | -------------- | ----------- |
| script1     | 20.24%          | 20.24%         | +0.00%      |
| script2     | 0.00%           | 1.20%          | **-1.20%**  |
| script3     | 5.49%           | 5.49%          | +0.00%      |
| **Average** | **8.58%**       | **8.98%**      | **-0.40%**  |

## Why Filtering Doesn't Help

### 1. Clean Audio Source

The scripted audio files are:

- Recorded in a controlled environment
- Already at optimal quality for Whisper
- Free from significant background noise
- Properly normalized

**Filtering clean audio adds no value and may introduce artifacts.**

### 2. Whisper's Robustness

Whisper models are:

- Trained on diverse, noisy audio
- Already robust to various audio conditions
- Designed to handle imperfect audio

**Whisper doesn't need pre-filtering for most use cases.**

### 3. Processing Overhead

The audio filter pipeline:

- Performs spectral analysis (FFT operations)
- Applies multiple filters in sequence
- Includes adaptive processing and monitoring
- Adds ~34 seconds per audio file

**The overhead far outweighs any potential benefit.**

## When Audio Filtering MIGHT Help

Audio filtering could be beneficial in these scenarios:

1. **Real-time microphone input** with:

   - Significant background noise (traffic, crowds, machinery)
   - Electrical interference (hum, buzz)
   - Echo or reverberation
   - Keyboard/mouse clicks during dictation

2. **Low-quality audio sources**:

   - Phone recordings with poor signal
   - Old recordings with tape hiss
   - Compressed audio with artifacts

3. **Specific noise patterns**:
   - Predictable, stationary noise (AC hum, fan noise)
   - Mechanical noise with harmonic patterns
   - Simple echo patterns

## Recommendations

### For Current Implementation

1. **Make filtering opt-in, not default**

   - Disable by default for file-based transcription
   - Enable only for real-time microphone input if needed

2. **Add audio quality detection**

   - Analyze input audio SNR (Signal-to-Noise Ratio)
   - Only apply filtering if SNR < threshold (e.g., 20 dB)
   - Skip filtering for clean audio automatically

3. **Optimize filter pipeline**

   - Reduce complexity for real-time use
   - Consider simpler filters (high-pass only)
   - Profile and optimize slow operations

4. **Add bypass mechanisms**
   - CLI flag: `--no-audio-filtering`
   - Config option: `enable_audio_filtering: false`
   - Automatic bypass for file inputs

### For Future Development

1. **Test with noisy audio**

   - Create test files with actual background noise
   - Test in real-world environments (office, street, home)
   - Measure improvement on genuinely noisy audio

2. **Benchmark against alternatives**

   - Compare with Whisper's built-in robustness
   - Test simpler preprocessing (normalization only)
   - Evaluate cost/benefit ratio

3. **Consider ML-based approaches**
   - Neural network-based noise reduction (e.g., RNNoise)
   - May be more effective than spectral methods
   - Could be faster with GPU acceleration

## Conclusion

**Audio filtering should be disabled by default for the current use case.** The scripted audio tests demonstrate that:

- Filtering provides **no accuracy improvement** on clean audio
- Processing time increases by **~10x** (unacceptable overhead)
- In one case, filtering **degraded** transcription quality

The audio filtering implementation is well-designed and may be valuable for noisy, real-time audio input. However, for file-based transcription of reasonably clean audio, it should be bypassed entirely.

### Action Items

1. ✅ Update test data README to document scripted audio files
2. ⚠️ Disable audio filtering by default in service configuration
3. ⚠️ Add CLI flag to enable filtering when needed
4. ⚠️ Implement automatic audio quality detection
5. 📋 Create noisy audio test suite for future validation
6. 📋 Profile and optimize filter pipeline for real-time use

## Test Methodology

### Test Setup

- **Whisper Model**: small (CPU, int8)
- **Audio Format**: WAV, 16kHz, mono
- **Filter Configuration**: Default AudioFilterPipeline settings
- **Measurement**: Word Error Rate (WER) using Levenshtein distance

### WER Calculation

WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

- Lower WER = better accuracy
- 0% WER = perfect transcription
- Negative improvement = filtering made it worse

### Test Files

All test files located in `tests/test_data/audio/scripted/`:

- `script1.wav` + `script1.txt` - Dates, numbers, times
- `script2.wav` + `script2.txt` - Punctuation, capitalization
- `script3.wav` + `script3.txt` - Homophones, technical vocabulary

### Running Tests

```bash
# Run full comparison test
pytest tests/test_speech_to_text/test_scripted_audio_comparison.py -v -s

# Run summary only
pytest tests/test_speech_to_text/test_scripted_audio_comparison.py::TestScriptedAudioComparison::test_all_scripts_summary -v -s

# Run individual script tests
pytest tests/test_speech_to_text/test_scripted_audio_comparison.py::TestScriptedAudioComparison::test_script1_comparison -v -s
```
