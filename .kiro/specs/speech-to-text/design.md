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
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Audio Buffer   │───▶│  Transcription  │
│   (Microphone)  │    │   & VAD         │    │   (Ollama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
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
4. **Transcription**: Processes complete speech segments when natural breaks are detected
5. **Output**: Returns transcribed text for each complete thought/sentence

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

**Purpose**: Uses faster-whisper library for local speech-to-text conversion

```python
class WhisperTranscriber:
    def __init__(self, model_size: str = "small")
    async def transcribe_audio(self, audio_data: bytes) -> str
    def is_model_available(self) -> bool
    def get_model_info(self) -> Dict[str, Any]
```

**Key Features:**

- Async transcription to prevent blocking
- Automatic GPU/CPU device selection
- Efficient faster-whisper implementation
- Audio format conversion for Ollama compatibility

### 4. SpeechToTextService Class

**Purpose**: Main orchestrator that coordinates all components

```python
class SpeechToTextService:
    def __init__(self)
    async def start_listening(self) -> None
    async def stop_listening(self) -> None
    def get_latest_transcription(self) -> Optional[str]
    def set_transcription_callback(self, callback: Callable[[str], None]) -> None
```

**Key Features:**

- Manages lifecycle of all components
- Provides callback mechanism for real-time transcription updates
- Handles graceful shutdown and error recovery
- Thread-safe access to transcription results

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
    confidence: float
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

## Model Selection Strategy

For an 8GB GPU setup, the optimal choice is **Whisper Small**:

- **Whisper Tiny (39MB)**: Fastest but lower accuracy, good for testing
- **Whisper Small (244MB)**: **Recommended** - Best balance of speed/accuracy for 8GB GPU
- **Whisper Medium (769MB)**: Higher accuracy, still fits in 8GB but slower
- **Whisper Large (1550MB)**: Best accuracy, may be tight on 8GB depending on other processes

The design uses Whisper Small as default but allows easy model switching for future GPU upgrades.

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

# Transcription Configuration
WHISPER_MODEL_SIZE = "small"  # Optimal for 8GB GPU
# Alternative models for future upgrades:
# WHISPER_MODEL_SIZE = "medium"  # For better accuracy with more GPU memory
# WHISPER_MODEL_SIZE = "large"   # For best accuracy with 16GB+ GPU
# Device selection is automatic (GPU if available, CPU fallback)
MAX_AUDIO_BUFFER_SIZE = 10  # seconds of audio
TRANSCRIPTION_TIMEOUT = 30  # seconds

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
