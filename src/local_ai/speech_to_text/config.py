"""Configuration constants for speech-to-text functionality."""

# Audio Configuration
DEFAULT_SAMPLE_RATE = 16000  # Hz, optimal for speech recognition
DEFAULT_CHUNK_SIZE = 1024  # samples per chunk
AUDIO_CHANNELS = 1  # mono audio

# Voice Activity Detection
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering
VAD_FRAME_DURATION = 30  # milliseconds
VAD_SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]  # Hz
VAD_SUPPORTED_FRAME_DURATIONS = [10, 20, 30]  # milliseconds

# Natural Break Detection
SHORT_PAUSE_THRESHOLD = 0.3  # seconds - normal speech rhythm pause
MEDIUM_PAUSE_THRESHOLD = 0.8  # seconds - likely sentence boundary
LONG_PAUSE_THRESHOLD = 2.0  # seconds - definite end of speech segment
MAX_SEGMENT_DURATION = 30.0  # seconds - force transcription of very long segments

# Adaptive Behavior
SILENCE_ADAPTATION_FACTOR = 0.1  # How quickly to adapt to speaker patterns
NOISE_COMPENSATION_THRESHOLD = 0.02  # Adjust VAD sensitivity based on background noise
PAUSE_HISTORY_SIZE = 20  # Number of recent pauses to track for adaptation
ADAPTIVE_PAUSE_MIN_SHORT = 0.2  # seconds - minimum short pause threshold
ADAPTIVE_PAUSE_MAX_SHORT = 0.5  # seconds - maximum short pause threshold
ADAPTIVE_PAUSE_MIN_MEDIUM = 0.5  # seconds - minimum medium pause threshold
ADAPTIVE_PAUSE_MAX_MEDIUM = 1.5  # seconds - maximum medium pause threshold

# Performance and Optimization
DEFAULT_BUFFER_SIZE = 160  # ~5 seconds at 30ms chunks
LATENCY_BUFFER_SIZE = 80  # ~2.5 seconds for low latency
ACCURACY_BUFFER_SIZE = 480  # ~15 seconds for high accuracy
HIGH_MEMORY_BUFFER_SIZE = 320  # ~10 seconds for systems with 8GB+ RAM

DEFAULT_MIN_SPEECH_DURATION = 0.5  # seconds
DEFAULT_MAX_SILENCE_DURATION = 2.0  # seconds
DEFAULT_MAX_AUDIO_BUFFER_SIZE = 10  # seconds
DEFAULT_TRANSCRIPTION_TIMEOUT = 30  # seconds

LATENCY_MIN_SPEECH_DURATION = 0.3  # seconds - shorter for responsiveness
LATENCY_MAX_SILENCE_DURATION = 1.0  # seconds - faster cutoff
LATENCY_CHUNK_SIZE = 512  # smaller chunks
LATENCY_VAD_FRAME_DURATION = 20  # milliseconds

ACCURACY_MIN_SPEECH_DURATION = 0.8  # seconds - longer for quality
ACCURACY_MAX_SILENCE_DURATION = 3.0  # seconds - allow longer pauses

RESOURCE_CHUNK_SIZE = 1024  # larger chunks for efficiency
RESOURCE_PROCESSING_INTERVAL = 0.02  # seconds - less frequent processing

# System Thresholds
HIGH_CPU_THRESHOLD = 80.0  # percent
HIGH_LATENCY_THRESHOLD = 5.0  # seconds
HIGH_MEMORY_GB = 8  # GB
HIGH_GPU_MEMORY_GB = 6  # GB
ULTRA_GPU_MEMORY_GB = 10  # GB

# Confidence Rating Configuration
CONFIDENCE_LOGPROB_MIN = -2.0  # Minimum expected avg_logprob value
CONFIDENCE_LOGPROB_MAX = -0.1  # Maximum expected avg_logprob value

# File and Data Limits
MAX_AUDIO_FILE_SIZE = 100 * 1024 * 1024  # 100MB
PERFORMANCE_HISTORY_SIZE = 10  # number of recent measurements to keep
OPTIMIZATION_CACHE_SIZE = 10  # number of cached configs per system

# Processing Intervals
ERROR_RECOVERY_SLEEP = 0.1  # seconds - sleep after processing errors
SILENCE_DURATION = 0.1  # seconds - for creating silence audio
VAD_SLOW_THRESHOLD = 0.1  # seconds - threshold for slow VAD detection

# Memory and Storage
BYTES_PER_KB = 1024
MINIMUM_PROCESSING_INTERVAL = 0.001  # seconds
MINIMUM_CHUNK_SIZE = 256  # samples
