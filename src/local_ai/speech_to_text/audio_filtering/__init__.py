"""Audio filtering and enhancement module for speech-to-text processing."""

from .adaptive_processor import AdaptiveProcessor
from .audio_filter_pipeline import AudioFilterPipeline
from .audio_normalizer import AudioNormalizer
from .interfaces import (
    AdaptiveProcessorInterface,
    AudioFilterInterface,
    AudioNormalizerInterface,
    NoiseReductionInterface,
    SpectralEnhancerInterface,
)
from .models import AudioProfile, FilterStats, NoiseType
from .noise_reduction import NoiseReductionEngine
from .performance_monitor import AudioFilterPerformanceMonitor
from .spectral_enhancer import SpectralEnhancer

__all__ = [
    "AudioProfile",
    "FilterStats",
    "NoiseType",
    "AudioFilterInterface",
    "NoiseReductionInterface",
    "AudioNormalizerInterface",
    "SpectralEnhancerInterface",
    "AdaptiveProcessorInterface",
    "AdaptiveProcessor",
    "NoiseReductionEngine",
    "AudioNormalizer",
    "SpectralEnhancer",
    "AudioFilterPipeline",
    "AudioFilterPerformanceMonitor",
]
