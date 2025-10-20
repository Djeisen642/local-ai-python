"""Data models for audio filtering and enhancement."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class NoiseType(Enum):
    """Types of noise that can be detected and filtered."""

    STATIONARY = "stationary"  # Constant background noise
    TRANSIENT = "transient"  # Keyboard, clicks, pops
    MECHANICAL = "mechanical"  # Fans, AC, machinery
    SPEECH = "speech"  # Background conversations
    MIXED = "mixed"  # Multiple noise types


@dataclass
class FilterStats:
    """Statistics about applied audio filtering."""

    noise_reduction_db: float
    signal_enhancement_db: float
    processing_latency_ms: float
    filters_applied: List[str]
    audio_quality_score: float


@dataclass
class AudioProfile:
    """Profile of audio characteristics for adaptive filtering."""

    snr_db: float
    dominant_frequencies: List[float]
    noise_type: NoiseType
    speech_presence: float
    recommended_filters: List[str]
