# Speech-to-Text Design Document

## Overview

The speech-to-text feature will implement real-time audio capture from a microphone and transcription using OpenAI's Whisper model running locally. The system will use a combination of audio processing libraries for microphone input and voice activity detection, paired with the faster-whisper library for efficient local speech recognition.

**Key Design Decisions:**

- Use PyAudio for cross-platform microphone access and real-time audio streaming
- Implement WebRTC VAD (Voice Activity Detection) for efficient speech detection
- Use faster-whisper library with Whisper Small model for optimal balance of quality, speed, and GPU memory usage on 8GB GPU
- Implement chunked audio processing to balance latency and accuracy
- Use asyncio for non-blocking audio processing and transcription

## Architecture

The system follows a producer-consumer pattern with three main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Audio Buffer   │───▶│  Transcription  │───▶│ Future Systems  │
│   (Microphone)  │    │   & VAD         │    │ (faster-whisper)│    │ (Embeddings,    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    │ Response, TTS)  │
         │                       │                       │            └─────────────────┘
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Audio Stream    │    │ Voice Activity  │    │ Text Output     │
│ Management      │    │ Detection       │    │ Handler         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Interaction Flow:

1. **Audio Input**: Continuously captures audio from microphone in small chunks
2. **Audio Buffer & VAD**: Buffers audio, detects speech activity, and tracks silence periods
3. **Natural Break Detection**: Identifies when user has finished speaking based on silence duration
4. **Transcription**: Processes complete speech segments using faster-whisper when natural breaks are detected
5. **Output**: Returns transcribed text for each complete thought/sentence
6. **Future Integration**: Transcribed text will be passed to downstream systems for:
   - Embedding generation and storage
   - AI response generation
   - Text-to-speech conversion

## Components and Interfaces

### 1. AudioCapture Class

**Purpose**: Manages microphone input and audio streaming

```python
class AudioCapture:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024)
    def start_capture(self) -> None
    def stop_capture(self) -> None
    def get_audio_chunk(self) -> Optional[bytes]
    def is_capturing(self) -> bool
```

**Key Features:**

- Configurable sample rate (default 16kHz for speech recognition)
- Non-blocking audio chunk retrieval
- Automatic microphone device selection
- Error handling for missing/unavailable microphones

### 2. VoiceActivityDetector Class

**Purpose**: Detects when speech is present in audio stream and identifies natural breaks

```python
class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30)
    def is_speech(self, audio_chunk: bytes) -> bool
    def get_speech_segments(self, audio_buffer: List[bytes]) -> List[bytes]
    def detect_speech_end(self, silence_duration: float) -> bool
    def reset_silence_timer(self) -> None
```

**Key Features:**

- WebRTC VAD implementation for robust speech detection
- Natural break detection through silence duration tracking
- Configurable sensitivity and silence thresholds through constants
- Handles background noise filtering
- Returns complete speech segments when natural breaks are detected

### 3. WhisperTranscriber Class

**Purpose**: Uses faster-whisper library with optimization and caching for local speech-to-text conversion

```python
class WhisperTranscriber:
    def __init__(self, model_size: str = "small")
    async def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult
    def is_model_available(self) -> bool
    def get_model_info(self) -> Dict[str, Any]
    def clear_model_cache(self) -> bool  # Clears HuggingFace cache
    def _calculate_confidence(self, segments: List[Segment]) -> float  # Convert avg_logprob to confidence
```

**Key Features:**

- Async transcription with optimized parameters (beam_size, temperature, etc.)
- Automatic GPU/CPU device selection with memory optimization
- Model caching with automatic cleanup and size management
- Performance optimization through compute_type and compression detection
- Cache management with reset capability
- Audio format conversion for Ollama compatibility
- **Confidence calculation from faster-whisper avg_logprob values**

### 4. SpeechToTextService Class

**Purpose**: Main orchestrator that coordinates all components and provides integration interface

```python
class SpeechToTextService:
    def __init__(self)
    async def start_listening(self) -> None
    async def stop_listening(self) -> None
    def get_latest_transcription(self) -> Optional[TranscriptionResult]
    def set_transcription_callback(self, callback: Callable[[TranscriptionResult], None]) -> None
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None
```

**Key Features:**

- Manages lifecycle of all components
- Provides callback mechanism for real-time transcription updates to downstream systems
- Handles graceful shutdown and error recovery
- Thread-safe access to transcription results with metadata
- Designed for integration with embedding, response generation, and TTS systems

## Data Models

### AudioChunk

```python
@dataclass
class AudioChunk:
    data: bytes
    timestamp: float
    sample_rate: int
    duration: float
```

### TranscriptionResult

```python
@dataclass
class TranscriptionResult:
    text: str
    confidence: float  # 0.0 to 1.0, derived from faster-whisper avg_logprob
    timestamp: float
    processing_time: float
```

### SpeechSegment

```python
@dataclass
class SpeechSegment:
    audio_data: bytes
    start_time: float
    end_time: float
    is_complete: bool
    silence_duration: float  # Duration of silence after speech

@dataclass
class SilenceState:
    is_silent: bool
    silence_start_time: float
    current_silence_duration: float
```

## Error Handling

### Microphone Errors

- **No microphone detected**: Graceful fallback with clear user message
- **Permission denied**: Instructions for granting microphone access
- **Device busy**: Retry logic with exponential backoff

### Whisper Model Errors

- **Model not downloaded**: Automatic download of Whisper model on first use
- **GPU memory insufficient**: Automatic fallback to CPU processing
- **Transcription timeout**: Chunk size adjustment and user notification

### Audio Processing Errors

- **Buffer overflow**: Automatic buffer management and oldest data dropping
- **Invalid audio format**: Automatic format conversion or error reporting
- **VAD failures**: Fallback to simple volume-based detection

## Testing Strategy

### Unit Tests

- **AudioCapture**: Mock PyAudio for testing audio stream management
- **VoiceActivityDetector**: Test with known speech/non-speech audio samples
- **WhisperTranscriber**: Mock faster-whisper library for transcription testing
- **SpeechToTextService**: Integration tests with mocked components

### Integration Tests

- **End-to-end audio flow**: Test complete pipeline with sample audio files
- **Real-time performance**: Measure latency from speech to transcription
- **Error scenarios**: Test graceful handling of various failure modes
- **Resource usage**: Monitor memory and CPU usage during extended operation

### Performance Tests

- **Latency measurement**: Target <2 second delay from speech end to transcription
- **Accuracy validation**: Test with various speakers and audio conditions
- **Resource monitoring**: Ensure reasonable CPU/memory usage
- **Concurrent operation**: Test multiple simultaneous transcription requests

## Natural Break Detection Strategy

The system implements intelligent speech segmentation to provide natural, conversational transcription:

### Silence-Based Segmentation

- **Short Pauses (0.3-0.8s)**: Considered normal speech rhythm, continue buffering
- **Medium Pauses (0.8-2.0s)**: Likely end of sentence/thought, trigger transcription
- **Long Pauses (>2.0s)**: Definite end of speech segment, finalize transcription

### Adaptive Thresholds

- **Dynamic Adjustment**: Silence thresholds adapt based on speaker patterns
- **Context Awareness**: Different thresholds for different types of speech (commands vs conversation)
- **Background Noise Compensation**: Adjust detection sensitivity based on ambient noise levels

### Speech Segment Management

- **Buffer Management**: Maintain rolling buffer of recent audio for context
- **Overlap Handling**: Ensure smooth transitions between segments
- **Incomplete Segment Recovery**: Handle interruptions and partial speech gracefully

## Model Selection and Optimization Strategy

### Model Selection

For an 8GB GPU setup, the optimal choice is **Whisper Small**:

- **Whisper Tiny (39MB)**: Fastest but lower accuracy, good for testing
- **Whisper Small (244MB)**: **Recommended** - Best balance of speed/accuracy for 8GB GPU
- **Whisper Medium (769MB)**: Higher accuracy, still fits in 8GB but slower
- **Whisper Large (1550MB)**: Best accuracy, may be tight on 8GB depending on other processes

### Performance Optimization

The system implements several optimization strategies:

**Memory Optimization:**

- Uses `float16` compute type to reduce GPU memory usage by ~50%
- Leverages faster-whisper's automatic model caching

**Processing Optimization:**

- Optimized beam search parameters (beam_size=5, best_of=5)
- Deterministic output with temperature=0.0 for consistent results
- Compression ratio detection to identify and filter repetitive transcriptions
- Log probability thresholds to filter low-confidence results

**Caching Strategy:**

_Model Cache (HuggingFace):_

- Whisper models automatically cached in `~/.cache/huggingface/` directory
- Managed by HuggingFace transformers library
- Reset with `--reset-model-cache` for corrupted model files

_Optimization Cache (System):_

- System capabilities, optimized configurations, and performance history
- Stored in `~/.cache/local_ai/speech_to_text/` directory
- Includes system fingerprinting for cache validation
- Reset with `--reset-optimization-cache` after system changes

## Confidence Rating Implementation

### Confidence Calculation Strategy

The system leverages faster-whisper's built-in confidence metrics to provide meaningful confidence ratings:

**Source Data from faster-whisper:**

- `avg_logprob`: Average log probability across all tokens in the transcription
- `no_speech_prob`: Probability that the audio contains no speech
- `compression_ratio`: Ratio indicating potential repetitive or nonsensical output

**Confidence Calculation:**

```python
def _calculate_confidence(self, segments: List[Segment]) -> float:
    """
    Convert faster-whisper avg_logprob to normalized confidence score (0.0-1.0)

    avg_logprob typically ranges from -2.0 (low confidence) to -0.1 (high confidence)
    We normalize this to a 0.0-1.0 scale for user-friendly display
    """
    if not segments:
        return 0.0

    # Calculate weighted average of segment confidences
    total_duration = sum(segment.end - segment.start for segment in segments)
    if total_duration == 0:
        return 0.0

    weighted_logprob = sum(
        segment.avg_logprob * (segment.end - segment.start)
        for segment in segments
    ) / total_duration

    # Convert log probability to confidence (0.0-1.0)
    # Typical range: avg_logprob from -2.0 to -0.1
    confidence = max(0.0, min(1.0, (weighted_logprob + 2.0) / 1.9))

    return confidence
```

### Confidence Display and Integration

The system displays confidence information and passes it to downstream systems:

**CLI Display Format:**

- "Transcribed text here (confidence: 85%)"
- "Another transcription (confidence: 42%)"

**Downstream Integration:**

- Full `TranscriptionResult` objects with confidence scores passed to callback handlers
- Downstream systems can use confidence values for their own decision-making

## Configuration Constants

```python
# Audio Configuration
SAMPLE_RATE = 16000  # Hz, optimal for speech recognition
CHUNK_SIZE = 1024    # samples per chunk
CHANNELS = 1         # mono audio
AUDIO_FORMAT = pyaudio.paInt16

# Voice Activity Detection
VAD_FRAME_DURATION = 30  # milliseconds
VAD_AGGRESSIVENESS = 2   # 0-3, higher = more aggressive filtering

# Model and Cache Configuration
WHISPER_MODEL_SIZE = "small"  # Optimal for 8GB GPU
COMPUTE_TYPE = "float16"  # Optimization for GPU memory efficiency
# Note: faster-whisper handles model caching automatically in ~/.cache/huggingface/

# Performance Optimization
BEAM_SIZE = 5  # Balance between accuracy and speed
BEST_OF = 5    # Number of candidates to consider
TEMPERATURE = 0.0  # Deterministic output (0.0) vs creative (1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4  # Detect repetitive transcriptions
LOGPROB_THRESHOLD = -1.0  # Filter low-confidence results
NO_SPEECH_THRESHOLD = 0.6  # Threshold for detecting silence

# Confidence Rating Configuration
CONFIDENCE_LOGPROB_MIN = -2.0  # Minimum expected avg_logprob value for normalization
CONFIDENCE_LOGPROB_MAX = -0.1  # Maximum expected avg_logprob value for normalization

# Buffer and Processing
MAX_AUDIO_BUFFER_SIZE = 10  # seconds of audio
TRANSCRIPTION_TIMEOUT = 30  # seconds
PROCESSING_CHUNK_SIZE = 30  # seconds - optimal chunk size for Whisper

# Natural Break Detection
MIN_SPEECH_DURATION = 0.5   # seconds - minimum speech to consider valid
SHORT_PAUSE_THRESHOLD = 0.3  # seconds - normal speech rhythm pause
MEDIUM_PAUSE_THRESHOLD = 0.8 # seconds - likely sentence boundary
LONG_PAUSE_THRESHOLD = 2.0   # seconds - definite end of speech segment
MAX_SEGMENT_DURATION = 30.0  # seconds - force transcription of very long segments

# Adaptive Behavior
SILENCE_ADAPTATION_FACTOR = 0.1  # How quickly to adapt to speaker patterns
NOISE_COMPENSATION_THRESHOLD = 0.02  # Adjust VAD sensitivity based on background noise
```

## Integration with Future Systems

The speech-to-text module is designed as the first component in a larger AI assistant pipeline:

### Output Format

- **Structured Output**: Transcription results include metadata (timestamp, confidence, processing time)
- **Clean Text**: Post-processed text suitable for downstream NLP systems
- **Segmented Results**: Natural speech segments that align with conversational boundaries

### Future System Integration Points

- **Embedding Pipeline**: Transcribed text will be processed for semantic embeddings and stored for context
- **Response Generation**: Clean transcription will be fed to AI models (potentially Ollama-based) for response generation
- **Text-to-Speech**: Generated responses will be converted back to speech for voice interaction
- **Context Management**: Speech segments will contribute to conversation history and user modeling

### API Design Considerations

- **Callback Architecture**: Supports real-time streaming to downstream systems
- **Async Processing**: Non-blocking design allows parallel processing in future components
- **Error Propagation**: Structured error handling for robust pipeline operation
- **Metadata Preservation**: Maintains timing and confidence information for downstream decision-making

## Command-Line Interface

The CLI provides a user-friendly interface with comprehensive argument support:

### Core Arguments

```bash
python -m local_ai.speech_to_text [OPTIONS]
```

### Available Options

**Essential Options:**

- `--help, -h` - Show help message and exit
- `--reset-model-cache` - Clear HuggingFace model cache and re-download models
- `--reset-optimization-cache` - Clear system optimization cache (capabilities, configs, performance)
- `--verbose, -v` - Enable verbose logging and debug information

### Usage Examples

```bash
# Basic usage with defaults
python -m local_ai.speech_to_text

# Reset model cache (useful if Whisper model is corrupted)
python -m local_ai.speech_to_text --reset-model-cache

# Reset optimization cache (useful after system changes or for troubleshooting)
python -m local_ai.speech_to_text --reset-optimization-cache

# Verbose mode for debugging
python -m local_ai.speech_to_text --verbose
```
