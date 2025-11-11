"""Configuration constants for speech-to-text functionality."""

# Audio Configuration
DEFAULT_SAMPLE_RATE = 16000  # Hz, optimal for speech recognition
DEFAULT_CHUNK_SIZE = 1024  # samples per chunk

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

# Audio Filtering Configuration
# Based on empirical testing for optimal noise reduction and speech preservation

# AudioFilterPipeline Configuration
AUDIO_FILTER_MAX_LATENCY_MS = 50.0  # Maximum allowed processing latency
AUDIO_FILTER_AGGRESSIVENESS = 0.5  # Overall filtering aggressiveness (0.0 to 1.0)
AUDIO_FILTER_MAX_FAILURES_PER_FILTER = 3  # Max failures before bypassing filter
AUDIO_FILTER_LATENCY_HISTORY_SIZE = 100  # Number of latency measurements to track

# Noise Reduction Engine Configuration
NOISE_REDUCTION_AGGRESSIVENESS = 0.5  # Noise reduction strength (0.0 to 1.0)
NOISE_REDUCTION_ALPHA = 2.0  # Base over-subtraction factor for spectral subtraction
NOISE_REDUCTION_BETA = 0.01  # Base spectral floor factor
NOISE_REDUCTION_ADAPTATION_RATE = 0.1  # Rate of noise profile adaptation
NOISE_REDUCTION_FFT_SIZE = 512  # FFT size for spectral analysis
NOISE_REDUCTION_HOP_SIZE = 128  # Hop size for overlap-add processing (FFT_SIZE // 4)

# Wiener Filter Configuration
WIENER_FILTER_ALPHA = 0.98  # Smoothing factor for Wiener coefficients
WIENER_SPEECH_PRESENCE_THRESHOLD = 0.5  # Threshold for speech presence detection
WIENER_MIN_SNR = 0.01  # Minimum SNR for Wiener filtering
WIENER_MAX_SNR = 100.0  # Maximum SNR for Wiener filtering

# Audio Normalization Configuration
AUDIO_NORM_TARGET_LEVEL_DB = -20.0  # Target RMS level in dB (optimal for Whisper)
AUDIO_NORM_MAX_GAIN_DB = 20.0  # Maximum gain in dB
AUDIO_NORM_COMPRESSION_RATIO = 4.0  # Dynamic range compression ratio
AUDIO_NORM_COMPRESSION_THRESHOLD_DB = -12.0  # Compression threshold in dB
AUDIO_NORM_ATTACK_TIME_SEC = 0.01  # AGC attack time (10ms)
AUDIO_NORM_RELEASE_TIME_SEC = 0.1  # AGC release time (100ms)
AUDIO_NORM_LIMITER_THRESHOLD = 0.95  # Peak limiter threshold (just below clipping)

# Spectral Enhancement Configuration
SPECTRAL_SPEECH_BAND_LOW_HZ = 300.0  # Lower bound of speech enhancement band
SPECTRAL_SPEECH_BAND_HIGH_HZ = 3400.0  # Upper bound of speech enhancement band
SPECTRAL_ENHANCEMENT_FACTOR = 1.5  # Speech band enhancement multiplier
SPECTRAL_HIGH_PASS_CUTOFF_HZ = 80.0  # Default high-pass filter cutoff
SPECTRAL_HIGH_PASS_ORDER = 4  # High-pass filter order
SPECTRAL_ECHO_FRAME_SIZE = 1024  # Frame size for echo reduction
SPECTRAL_ECHO_HOP_SIZE = 512  # Hop size for echo reduction
SPECTRAL_TRANSIENT_FRAME_SIZE = 512  # Frame size for transient suppression
SPECTRAL_TRANSIENT_HOP_SIZE = 256  # Hop size for transient suppression
SPECTRAL_TRANSIENT_THRESHOLD = 2.0  # Threshold for transient detection

# Adaptive Processor Configuration
ADAPTIVE_FFT_SIZE = 1024  # FFT size for audio analysis
ADAPTIVE_SPEECH_FREQ_LOW_HZ = 80.0  # Lower bound of speech frequency range
ADAPTIVE_SPEECH_FREQ_HIGH_HZ = 4000.0  # Upper bound of speech frequency range
ADAPTIVE_FUNDAMENTAL_FREQ_LOW_HZ = 80.0  # Lower bound of fundamental frequency
ADAPTIVE_FUNDAMENTAL_FREQ_HIGH_HZ = 400.0  # Upper bound of fundamental frequency
ADAPTIVE_STATIONARITY_THRESHOLD = 0.8  # Threshold for stationary noise detection
ADAPTIVE_TRANSIENT_PEAK_RATIO_THRESHOLD = 10.0  # Peak-to-average ratio for transients
ADAPTIVE_SPEECH_PRESENCE_THRESHOLD = 0.5  # Threshold for speech presence
ADAPTIVE_HARMONIC_THRESHOLD = 0.7  # Threshold for harmonic content detection
ADAPTIVE_LEARNING_RATE = 0.1  # Learning rate for parameter adaptation
ADAPTIVE_ADAPTATION_THRESHOLD = 0.05  # Threshold for triggering adaptation

# SNR Thresholds for Filter Selection
SNR_THRESHOLD_HIGH_DB = 2.0  # High SNR threshold (minimal processing)
SNR_THRESHOLD_MEDIUM_DB = 0.5  # Medium SNR threshold (moderate processing)
SNR_THRESHOLD_LOW_DB = -5.0  # Low SNR threshold (aggressive processing)

# Noise Type Detection Thresholds
NOISE_STATIONARITY_THRESHOLD = 0.8  # Threshold for stationary noise classification
NOISE_TRANSIENT_THRESHOLD = 10.0  # Peak-to-average ratio for transient noise
NOISE_MECHANICAL_HARMONIC_THRESHOLD = 3  # Number of harmonics for mechanical noise
NOISE_SPEECH_CHARACTERISTICS_THRESHOLD = 0.6  # Threshold for speech-like characteristics

# Performance Optimization Thresholds
FILTER_CPU_USAGE_THRESHOLD = 80.0  # CPU usage threshold for complexity reduction
FILTER_MEMORY_USAGE_THRESHOLD_MB = 100.0  # Memory usage threshold in MB
FILTER_LATENCY_WARNING_THRESHOLD_MS = 40.0  # Latency warning threshold
FILTER_LATENCY_CRITICAL_THRESHOLD_MS = 80.0  # Latency critical threshold
FILTER_PROCESSING_TIMEOUT_SEC = 1.0  # Maximum time for filter processing

# Buffer Management Configuration
FILTER_AUDIO_BUFFER_SIZE = 4096  # Audio buffer size for filtering
FILTER_OVERLAP_SIZE = 1024  # Overlap size for windowed processing
FILTER_WINDOW_TYPE = "hann"  # Window function type for spectral processing
FILTER_MIN_FRAME_SIZE = 256  # Minimum frame size for processing
FILTER_MAX_FRAME_SIZE = 2048  # Maximum frame size for processing

# Quality and Effectiveness Metrics
FILTER_QUALITY_HISTORY_SIZE = 50  # Number of quality measurements to track
FILTER_EFFECTIVENESS_THRESHOLD = 0.7  # Minimum effectiveness for filter success
FILTER_NOISE_REDUCTION_TARGET_DB = 10.0  # Target noise reduction in dB
FILTER_SIGNAL_ENHANCEMENT_TARGET_DB = 2.0  # Target signal enhancement in dB
