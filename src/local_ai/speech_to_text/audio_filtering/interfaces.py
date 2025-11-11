"""Abstract interfaces for audio filtering components."""

from abc import ABC, abstractmethod

import numpy as np

from ..models import AudioChunk
from .models import AudioProfile, FilterStats, NoiseType


class AudioFilterInterface(ABC):
    """Base interface for all audio filtering components."""

    @abstractmethod
    async def process_audio_chunk(self, audio_chunk: AudioChunk) -> AudioChunk:
        """
        Process an audio chunk and return the filtered result.

        Args:
            audio_chunk: Input audio chunk to process

        Returns:
            Processed audio chunk with filtering applied
        """
        pass

    @abstractmethod
    def get_filter_stats(self) -> FilterStats:
        """
        Get statistics about the filtering performance.

        Returns:
            FilterStats containing performance metrics
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the filter state and adaptive parameters."""
        pass


class NoiseReductionInterface(ABC):
    """Interface for noise reduction engines."""

    @abstractmethod
    def update_noise_profile(self, audio_data: np.ndarray) -> None:
        """
        Update the noise profile based on audio data.

        Args:
            audio_data: Audio data to analyze for noise characteristics
        """
        pass

    @abstractmethod
    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Noise-reduced audio data
        """
        pass

    @abstractmethod
    def detect_noise_type(self, audio_data: np.ndarray) -> NoiseType:
        """
        Detect the type of noise present in the audio.

        Args:
            audio_data: Audio data to analyze

        Returns:
            Detected noise type
        """
        pass

    @abstractmethod
    def get_noise_reduction_db(self) -> float:
        """
        Get the current noise reduction level in dB.

        Returns:
            Noise reduction level in decibels
        """
        pass


class AudioNormalizerInterface(ABC):
    """Interface for audio normalization and gain control."""

    @abstractmethod
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio levels for optimal processing.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Normalized audio data
        """
        pass

    @abstractmethod
    def apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply automatic gain control to audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with AGC applied
        """
        pass

    @abstractmethod
    def compress_dynamic_range(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Compressed audio data
        """
        pass

    @abstractmethod
    def get_current_level(self) -> float:
        """
        Get the current audio level.

        Returns:
            Current audio level in dB
        """
        pass


class SpectralEnhancerInterface(ABC):
    """Interface for spectral enhancement and frequency domain processing."""

    @abstractmethod
    def enhance_speech_frequencies(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance speech frequency bands for better clarity.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with enhanced speech frequencies
        """
        pass

    @abstractmethod
    def apply_high_pass_filter(
        self, audio_data: np.ndarray, cutoff: float = 80.0
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.

        Args:
            audio_data: Input audio data as numpy array
            cutoff: High-pass filter cutoff frequency in Hz

        Returns:
            High-pass filtered audio data
        """
        pass

    @abstractmethod
    def reduce_echo(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Reduce echo and reverberation in audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with reduced echo
        """
        pass

    @abstractmethod
    def suppress_transients(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Suppress transient noises like keyboard clicks and pops.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with suppressed transients
        """
        pass


class AdaptiveProcessorInterface(ABC):
    """Interface for adaptive audio processing intelligence."""

    @abstractmethod
    def analyze_audio_characteristics(self, audio_data: np.ndarray) -> AudioProfile:
        """
        Analyze audio characteristics and create a profile.

        Args:
            audio_data: Audio data to analyze

        Returns:
            AudioProfile containing analysis results
        """
        pass

    @abstractmethod
    def select_optimal_filters(self, profile: AudioProfile) -> list[str]:
        """
        Select optimal filters based on audio profile.

        Args:
            profile: Audio profile from analysis

        Returns:
            List of recommended filter names
        """
        pass

    @abstractmethod
    def update_processing_parameters(self, effectiveness: float) -> None:
        """
        Update processing parameters based on effectiveness feedback.

        Args:
            effectiveness: Effectiveness score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def get_processing_recommendations(self) -> dict[str, float]:
        """
        Get processing parameter recommendations.

        Returns:
            Dictionary of parameter names and recommended values
        """
        pass
