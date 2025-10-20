"""Audio filtering and enhancement module for speech-to-text processing."""

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
    "NoiseReductionEngine",
    "AudioNormalizer",
    "SpectralEnhancer",
]
