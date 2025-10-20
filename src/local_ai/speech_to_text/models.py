"""Data models for speech-to-text functionality."""

from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata."""

    data: bytes
    timestamp: float
    sample_rate: int
    duration: float

    # Audio filtering metadata
    noise_level: float = 0.0
    signal_level: float = 0.0
    snr_db: float = 0.0
    is_filtered: bool = False


@dataclass
class TranscriptionResult:
    """Represents the result of speech transcription."""

    text: str
    confidence: float
    timestamp: float
    processing_time: float


@dataclass
class SpeechSegment:
    """Represents a segment of speech audio."""

    audio_data: bytes
    start_time: float
    end_time: float
    is_complete: bool
