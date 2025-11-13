"""Configuration constants for speech-to-text functionality."""

from pathlib import Path

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
LATENCY_CHUNK_SIZE = 480  # Aligned with 30ms VAD frame at 16kHz (480 samples = 960 bytes)
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

# Audio Capture Constants
AUDIO_LEVEL_LOG_INTERVAL = 5.0  # seconds - log audio levels every 5 seconds
AUDIO_LEVEL_THRESHOLD = 0.01  # Threshold for "significant" audio activity
AUDIO_SAMPLE_MAX_VALUE = 32767.0  # Maximum value for 16-bit audio samples
AUDIO_SAMPLE_NORMALIZATION = 32768.0  # Normalization factor for 16-bit audio

# VAD Constants
VAD_MIN_SAMPLES_FOR_ADAPTATION = 3  # Minimum samples needed for threshold adaptation
VAD_ADAPTIVE_MULTIPLIER_LONG = 1.2  # Multiplier for long pause adaptation
VAD_ADAPTIVE_MULTIPLIER_SHORT = 0.8  # Multiplier for short pause adaptation
VAD_ADAPTIVE_OFFSET = 0.5  # Offset for adaptive long pause threshold
VAD_DEBUG_LOG_INTERVAL = 10.0  # seconds - log VAD stats every 10 seconds

# Transcriber Constants
TRANSCRIBER_CHANNELS_MONO = 1  # Mono audio channel count
TRANSCRIBER_SAMPLE_WIDTH = 2  # 16-bit audio sample width in bytes
TRANSCRIBER_SEGMENT_RATIO_LOW = 0.3  # Low signal ratio threshold
TRANSCRIBER_SEGMENT_RATIO_MEDIUM = 0.5  # Medium signal ratio threshold

# Service Constants
SERVICE_TRANSCRIPTION_WINDOW = 60.0  # seconds - window for transcription stats
SERVICE_PERFORMANCE_WINDOW = 300.0  # seconds - window for performance stats (5 minutes)
SERVICE_HIGH_LATENCY_WARNING = 5.0  # seconds - latency threshold for warnings

# Optimization Constants
OPT_MIN_CPU_CORES = 1  # Minimum CPU core count
OPT_FAST_CPU_CORES = 4  # CPU cores for fast processing
OPT_MANY_CPU_CORES = 8  # CPU cores for many-core optimization
OPT_PROCESSING_INTERVAL_FAST = 0.005  # seconds - fast processing interval
OPT_PROCESSING_INTERVAL_NORMAL = 0.01  # seconds - normal processing interval
OPT_PROCESSING_INTERVAL_SLOW = 0.02  # seconds - slow processing interval
OPT_MAX_AUDIO_BUFFER_LARGE = 15  # seconds - large audio buffer
OPT_MAX_AUDIO_BUFFER_SMALL = 5  # seconds - small audio buffer
OPT_VAD_AGGRESSIVENESS_BOOST = 1  # Boost for VAD aggressiveness
OPT_VAD_AGGRESSIVENESS_REDUCE = 1  # Reduction for VAD aggressiveness
OPT_VAD_AGGRESSIVENESS_MAX = 3  # Maximum VAD aggressiveness
OPT_VAD_AGGRESSIVENESS_MIN = 1  # Minimum VAD aggressiveness
OPT_CHUNK_SIZE_DIVISOR = 2  # Divisor for chunk size reduction
OPT_INTERVAL_DIVISOR = 2  # Divisor for interval reduction
OPT_INTERVAL_MULTIPLIER = 2  # Multiplier for interval increase
OPT_SILENCE_REDUCTION = 0.5  # seconds - silence duration reduction
OPT_CHUNK_SIZE_MULTIPLIER = 2  # Multiplier for chunk size increase
OPT_ADAPTATION_COUNT_THRESHOLD = 3  # Minimum adaptations before changes
OPT_PERFORMANCE_HISTORY_MIN = 3  # Minimum performance history for adaptation
OPT_PERFORMANCE_HISTORY_RECENT = 5  # Recent performance history window

# Performance Monitor Constants
PERF_TRANSCRIPTION_LATENCY_HIGH = 5.0  # seconds - high latency threshold
PERF_SUCCESS_RATE_LOW = 0.9  # Minimum acceptable success rate (90%)
PERF_STATS_WINDOW_SHORT = 60.0  # seconds - short stats window (1 minute)
PERF_STATS_WINDOW_LONG = 300.0  # seconds - long stats window (5 minutes)

# Audio Filtering - Noise Reduction Constants
NR_ALPHA_BASE = 2.0  # Base over-subtraction factor
NR_ALPHA_MULTIPLIER = 2.0  # Multiplier for aggressiveness adjustment
NR_BETA_BASE = 0.01  # Base spectral floor factor
NR_BETA_MULTIPLIER = 0.04  # Multiplier for aggressiveness adjustment
NR_WIENER_ALPHA = 0.98  # Smoothing factor for Wiener coefficients
NR_SPEECH_PRESENCE_THRESHOLD = 0.5  # Threshold for speech presence
NR_SNR_MIN = 0.01  # Minimum SNR for Wiener filtering
NR_SNR_MAX = 100.0  # Maximum SNR for Wiener filtering
NR_FFT_SIZE = 512  # FFT size for spectral analysis
NR_HOP_SIZE_DIVISOR = 4  # Divisor for hop size calculation
NR_STATIONARITY_THRESHOLD = 0.8  # Threshold for stationary noise
NR_TRANSIENT_THRESHOLD = 10.0  # Peak-to-average ratio for transient noise
NR_MECHANICAL_HARMONIC_THRESHOLD = 3  # Number of harmonics for mechanical noise
NR_PEAK_HEIGHT_FACTOR = 0.1  # Factor for peak detection height
NR_HARMONIC_RATIO_TOLERANCE = 0.1  # Tolerance for harmonic ratio matching
NR_SPEECH_RATIO_THRESHOLD = 0.6  # Threshold for speech characteristics
NR_SEGMENT_COUNT = 4  # Number of segments for stationarity analysis
NR_SEGMENT_MIN_SIZE = 100  # Minimum segment size for analysis
NR_NOISE_PERCENTILE = 25  # Percentile for noise level estimation (25%)
NR_SIGNAL_PERCENTILE = 75  # Percentile for signal level estimation (75%)

# Audio Filtering - Spectral Enhancement Constants
SE_HIGH_PASS_ORDER = 4  # High-pass filter order
SE_SPEECH_BAND_LOW = 300.0  # Hz - lower bound of speech band
SE_SPEECH_BAND_HIGH = 3400.0  # Hz - upper bound of speech band
SE_ENHANCEMENT_FACTOR = 1.5  # Speech band enhancement multiplier
SE_TRANSITION_WIDTH = 200.0  # Hz - transition zone width
SE_FRAME_SIZE = 1024  # Frame size for processing
SE_HOP_SIZE = 512  # Hop size for overlap-add
SE_ECHO_FRAME_SIZE = 1024  # Frame size for echo reduction
SE_ECHO_HOP_SIZE = 512  # Hop size for echo reduction
SE_ECHO_THRESHOLD = 0.3  # Threshold for echo detection
SE_ECHO_PROFILE_FRAMES = 5  # Number of frames for echo profile
SE_ECHO_ALPHA = 0.5  # Over-subtraction factor for echo
SE_ECHO_BETA = 0.3  # Spectral floor for echo reduction
SE_TRANSIENT_FRAME_SIZE = 512  # Frame size for transient suppression
SE_TRANSIENT_HOP_SIZE = 256  # Hop size for transient suppression
SE_TRANSIENT_THRESHOLD = 2.0  # Threshold for transient detection
SE_TRANSIENT_SUPPRESSION_SPEECH = 0.7  # Suppression factor for speech band
SE_TRANSIENT_SUPPRESSION_HIGH = 0.4  # Suppression factor for high frequencies
SE_TRANSIENT_HF_ROLLOFF = 0.95  # High-frequency rolloff factor
SE_TRANSIENT_HF_CUTOFF = 4000.0  # Hz - high-frequency cutoff
SE_TRANSIENT_HF_CUTOFF_DETECT = 2000.0  # Hz - high-frequency detection cutoff
SE_SPECTRAL_FLOOR_FACTOR = 0.2  # Spectral floor factor for transient suppression

# Audio Filtering - Normalization Constants
NORM_TARGET_LEVEL = -20.0  # dB - target RMS level
NORM_MAX_GAIN = 20.0  # dB - maximum gain
NORM_COMPRESSION_RATIO = 4.0  # Compression ratio
NORM_COMPRESSION_THRESHOLD = -12.0  # dB - compression threshold
NORM_ATTACK_TIME = 0.01  # seconds - attack time (10ms)
NORM_RELEASE_TIME = 0.1  # seconds - release time (100ms)
NORM_LIMITER_THRESHOLD = 0.95  # Peak limiter threshold
NORM_LIMITER_RELEASE = 0.01  # seconds - limiter release time (10ms)
NORM_INITIAL_LEVEL = -60.0  # dB - initial level estimate
NORM_INITIAL_GAIN = 1.0  # Initial gain value
NORM_CLIPPING_THRESHOLD = 0.95  # Threshold for clipping detection

# Audio Filtering - Adaptive Processor Constants
AP_FFT_SIZE = 1024  # FFT size for analysis
AP_HOP_SIZE_DIVISOR = 4  # Divisor for hop size calculation
AP_SPEECH_FREQ_LOW = 80.0  # Hz - lower bound of speech frequency range
AP_SPEECH_FREQ_HIGH = 4000.0  # Hz - upper bound of speech frequency range
AP_FORMANT_F1 = 700.0  # Hz - first formant frequency
AP_FORMANT_F2 = 1220.0  # Hz - second formant frequency
AP_FORMANT_F3 = 2600.0  # Hz - third formant frequency
AP_FUNDAMENTAL_LOW = 80.0  # Hz - lower bound of fundamental frequency
AP_FUNDAMENTAL_HIGH = 400.0  # Hz - upper bound of fundamental frequency
AP_STATIONARITY_THRESHOLD = 0.8  # Threshold for stationary noise
AP_TRANSIENT_PEAK_RATIO = 10.0  # Peak-to-average ratio for transients
AP_TRANSIENT_PEAK_RATIO_HIGH = 20.0  # High peak-to-average ratio threshold
AP_SPEECH_PRESENCE_THRESHOLD = 0.5  # Threshold for speech presence
AP_HARMONIC_THRESHOLD = 0.7  # Threshold for harmonic content
AP_LEARNING_RATE = 0.1  # Learning rate for adaptation
AP_ADAPTATION_THRESHOLD = 0.05  # Threshold for triggering adaptation
AP_SNR_HIGH = 2.0  # dB - high SNR threshold
AP_SNR_MEDIUM = 0.5  # dB - medium SNR threshold
AP_SNR_LOW = -5.0  # dB - low SNR threshold
AP_FORMANT_TOLERANCE = 300.0  # Hz - tolerance for formant matching
AP_FORMANT_WEIGHT = 0.5  # Weight for formant matching in speech score
AP_FORMANT_BONUS = 0.2  # Bonus for multiple formant matches
AP_FUNDAMENTAL_WEIGHT = 0.2  # Weight for fundamental frequency in speech score
AP_SPEECH_LIKELIHOOD_WEIGHT = 0.6  # Weight for speech likelihood in speech score
AP_SPEECH_RANGE_BONUS_HIGH = 0.25  # Bonus for many frequencies in speech range
AP_SPEECH_RANGE_BONUS_MED = 0.2  # Bonus for medium frequencies in speech range
AP_SPEECH_RANGE_BONUS_LOW = 0.15  # Bonus for few frequencies in speech range
AP_HARMONIC_PENALTY = 0.7  # Penalty for harmonic patterns
AP_HARMONIC_PENALTY_STRONG = 0.15  # Strong penalty for mechanical harmonics
AP_SYNTHETIC_SPEECH_BONUS = 0.1  # Bonus for synthetic speech characteristics
AP_TRANSIENT_PENALTY = 0.1  # Penalty for transient characteristics
AP_SPEECH_LIKELIHOOD_LOW = 0.4  # Low speech likelihood threshold
AP_SPEECH_LIKELIHOOD_MED = 0.5  # Medium speech likelihood threshold
AP_SPEECH_LIKELIHOOD_HIGH = 0.7  # High speech likelihood threshold
AP_SPEECH_PRESENCE_HIGH = 0.8  # High speech presence threshold
AP_HARMONIC_RATIO_TOLERANCE = 0.2  # Tolerance for harmonic ratio matching
AP_HARMONIC_RATIO_MIN = 1.4  # Minimum harmonic ratio
AP_HARMONIC_RATIO_MAX = 4.0  # Maximum harmonic ratio
AP_MECHANICAL_FREQ_THRESHOLD = 500.0  # Hz - threshold for mechanical noise frequency
AP_STATIONARITY_DEFAULT = 0.5  # Default stationarity for short signals
AP_SEGMENT_COUNT = 4  # Number of segments for analysis
AP_SEGMENT_MIN_SIZE = 1024  # Minimum segment size for analysis
AP_CORRELATION_LAG_MIN = 20  # Minimum lag for autocorrelation
AP_PEAK_HEIGHT_FACTOR = 0.3  # Factor for peak detection height
AP_PEAK_HEIGHT_FACTOR_LOW = 0.1  # Low factor for peak detection
AP_PEAK_HEIGHT_FACTOR_VERYLOW = 0.001  # Very low factor for peak detection
AP_PEAK_DISTANCE_DIVISOR = 100  # Divisor for peak distance calculation
AP_DOMINANT_FREQ_COUNT = 10  # Number of dominant frequencies to return
AP_FREQ_MIN = 10.0  # Hz - minimum valid frequency
AP_CENTROID_RANGE_OPTIMAL = (300.0, 2500.0)  # Hz - optimal centroid range
AP_CENTROID_RANGE_ACCEPTABLE = (100.0, 4000.0)  # Hz - acceptable centroid range
AP_CENTROID_SCORE_OPTIMAL = 1.0  # Score for optimal centroid
AP_CENTROID_SCORE_ACCEPTABLE = 0.7  # Score for acceptable centroid
AP_CENTROID_SCORE_POOR = 0.2  # Score for poor centroid
AP_ROLLOFF_PERCENTILE = 0.85  # Percentile for spectral rolloff
AP_ROLLOFF_RANGE_OPTIMAL = (1500.0, 6000.0)  # Hz - optimal rolloff range
AP_ROLLOFF_RANGE_ACCEPTABLE = (800.0, 8000.0)  # Hz - acceptable rolloff range
AP_ROLLOFF_SCORE_OPTIMAL = 1.0  # Score for optimal rolloff
AP_ROLLOFF_SCORE_ACCEPTABLE = 0.7  # Score for acceptable rolloff
AP_ROLLOFF_SCORE_POOR = 0.3  # Score for poor rolloff
AP_PEAK_SCORE_DIVISOR = 5.0  # Divisor for peak score normalization
AP_CENTROID_WEIGHT = 0.4  # Weight for centroid in speech likelihood
AP_ROLLOFF_WEIGHT = 0.4  # Weight for rolloff in speech likelihood
AP_PEAK_WEIGHT = 0.2  # Weight for peaks in speech likelihood
AP_EFFECTIVENESS_HISTORY_SIZE = 100  # Number of effectiveness samples to keep
AP_AGGRESSIVENESS_INCREASE = 0.1  # Increase in aggressiveness for poor performance
AP_AGGRESSIVENESS_DECREASE = 0.05  # Decrease in aggressiveness for good performance
AP_AGGRESSIVENESS_MAX = 0.5  # Maximum aggressiveness adjustment
AP_AGGRESSIVENESS_MIN = -0.2  # Minimum aggressiveness adjustment
AP_THRESHOLD_ADJUSTMENT_POOR = 0.95  # Adjustment factor for poor performance
AP_THRESHOLD_ADJUSTMENT_GOOD = 1.02  # Adjustment factor for good performance
AP_THRESHOLD_MAX = 0.8  # Maximum threshold value

# Audio Filtering - Pipeline Constants
AFP_QUALITY_BASELINE = 0.8  # Baseline quality score for unprocessed audio
AFP_QUALITY_NEUTRAL = 0.7  # Neutral quality score on error
AFP_CLIPPING_THRESHOLD = 0.95  # Threshold for clipping detection
AFP_CLIPPING_PENALTY = 0.2  # Penalty for clipping
AFP_SIGNAL_LOSS_THRESHOLD_LOW = 0.3  # Low signal loss threshold
AFP_SIGNAL_LOSS_THRESHOLD_MED = 0.5  # Medium signal loss threshold
AFP_SIGNAL_LOSS_PENALTY_HIGH = 0.3  # High penalty for signal loss
AFP_SIGNAL_LOSS_PENALTY_MED = 0.1  # Medium penalty for signal loss
AFP_QUALITY_BASE = 0.9  # Base quality score
AFP_AGGRESSIVENESS_CHANGE_THRESHOLD = 0.1  # Threshold for aggressiveness change

# Optimization - System Detection Constants
OPT_DEFAULT_MEMORY_GB = 4  # Default conservative memory estimate in GB
OPT_DEFAULT_GPU_MEMORY_GB = 0  # Default GPU memory when unavailable

# Optimization - Model Selection Constants
OPT_MODEL_SIZE_DEFAULT = "small"  # Default Whisper model size
OPT_MODEL_SIZE_LARGE = "large"  # Large Whisper model
OPT_MODEL_SIZE_MEDIUM = "medium"  # Medium Whisper model
OPT_MODEL_SIZE_SMALL = "small"  # Small Whisper model
OPT_MODEL_SIZE_TINY = "tiny"  # Tiny Whisper model
OPT_USE_ENGLISH_ONLY_MODEL = (
    True  # Use English-only models (.en suffix) for better performance
)
OPT_USE_DISTILLED_MODELS = (
    True  # Use distilled models for better speed/accuracy tradeoff (experimental)
)

# Optimization - Compute Type Constants
OPT_COMPUTE_TYPE_INT8 = "int8"  # INT8 compute type (CPU optimized)
OPT_COMPUTE_TYPE_FLOAT16 = "float16"  # Float16 compute type (GPU optimized)

# Optimization - Device Constants
OPT_DEVICE_CPU = "cpu"  # CPU device
OPT_DEVICE_CUDA = "cuda"  # CUDA/GPU device

# Optimization - Platform Constants
OPT_PLATFORM_LINUX = "Linux"  # Linux platform identifier

# Optimization - Configuration Constants
OPT_PROCESSING_INTERVAL_DEFAULT = 0.01  # seconds - default processing interval
OPT_PROCESSING_INTERVAL_FAST_CPU = 0.005  # seconds - fast processing for 4+ cores
OPT_PROCESSING_INTERVAL_LATENCY = 0.005  # seconds - latency-optimized processing
OPT_PROCESSING_INTERVAL_ADAPTIVE_MAX = 0.05  # seconds - max adaptive processing interval
OPT_MAX_CONCURRENT_TRANSCRIPTIONS = 1  # Default max concurrent transcriptions
OPT_MAX_AUDIO_BUFFER_HIGH_MEM = 15  # seconds - buffer for high memory systems
OPT_MAX_AUDIO_BUFFER_RESOURCE = 5  # seconds - buffer for resource-constrained systems

# Optimization - CPU Core Thresholds
OPT_CPU_CORES_FAST = 4  # CPU cores threshold for fast processing
OPT_CPU_CORES_MANY = 8  # CPU cores threshold for many-core optimizations

# Optimization - VAD Aggressiveness Adjustments
OPT_VAD_AGGRESSIVENESS_LINUX_BOOST = 1  # Boost for Linux platform
OPT_VAD_AGGRESSIVENESS_ACCURACY_REDUCE = 1  # Reduction for accuracy mode

# Optimization - Adaptive Optimizer Constants
OPT_ADAPTIVE_MIN_HISTORY = 3  # Minimum history entries for adaptation
OPT_ADAPTIVE_RECENT_WINDOW = 3  # Recent performance window size
OPT_ADAPTIVE_LATENCY_THRESHOLD = 5.0  # seconds - high latency threshold
OPT_ADAPTIVE_SILENCE_REDUCTION = 0.5  # seconds - silence duration reduction
OPT_ADAPTIVE_INTERVAL_MAX = 0.05  # seconds - maximum processing interval

# Audio Debugging Configuration
AUDIO_DEBUG_ENABLED = False  # Default disabled
AUDIO_DEBUG_DEFAULT_DIR = (
    Path.home() / ".cache" / "local_ai" / "audio_debug"
)  # Default output directory
AUDIO_DEBUG_FILENAME_FORMAT = "audio_{date}_{time}_{duration_ms}.wav"  # Filename format
