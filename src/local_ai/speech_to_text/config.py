"""Configuration constants for speech-to-text functionality."""

# Audio Configuration
SAMPLE_RATE = 16000  # Hz, optimal for speech recognition
CHUNK_SIZE = 1024    # samples per chunk
CHANNELS = 1         # mono audio

# Voice Activity Detection
VAD_FRAME_DURATION = 30  # milliseconds
VAD_AGGRESSIVENESS = 2   # 0-3, higher = more aggressive filtering

# Transcription Configuration
WHISPER_MODEL_SIZE = "small"  # Optimal for 8GB GPU
MAX_AUDIO_BUFFER_SIZE = 10  # seconds of audio
TRANSCRIPTION_TIMEOUT = 30  # seconds

# Performance Tuning
MIN_SPEECH_DURATION = 0.5   # seconds
MAX_SILENCE_DURATION = 2.0  # seconds before finalizing transcription

# Performance Optimization Presets
OPTIMIZATION_PRESETS = {
    "latency": {
        "description": "Optimized for low latency response",
        "chunk_size": 512,
        "processing_interval": 0.005,
        "min_speech_duration": 0.3,
        "max_silence_duration": 1.0,
        "vad_frame_duration": 20,
        "buffer_size": 80,  # ~2.5 seconds
        "whisper_model_size": "small"
    },
    "accuracy": {
        "description": "Optimized for transcription accuracy",
        "chunk_size": 1024,
        "processing_interval": 0.01,
        "min_speech_duration": 0.8,
        "max_silence_duration": 3.0,
        "vad_frame_duration": 30,
        "buffer_size": 480,  # ~15 seconds
        "whisper_model_size": "medium"
    },
    "resource": {
        "description": "Optimized for low resource usage",
        "chunk_size": 2048,
        "processing_interval": 0.02,
        "min_speech_duration": 0.5,
        "max_silence_duration": 2.0,
        "vad_frame_duration": 30,
        "buffer_size": 160,  # ~5 seconds
        "whisper_model_size": "tiny"
    },
    "balanced": {
        "description": "Balanced performance and resource usage",
        "chunk_size": 1024,
        "processing_interval": 0.01,
        "min_speech_duration": 0.5,
        "max_silence_duration": 2.0,
        "vad_frame_duration": 30,
        "buffer_size": 160,  # ~5 seconds
        "whisper_model_size": "small"
    }
}

# System-specific optimization thresholds
SYSTEM_OPTIMIZATION_THRESHOLDS = {
    "high_cpu_cores": 4,
    "high_memory_gb": 8,
    "gpu_memory_gb": 4
}
