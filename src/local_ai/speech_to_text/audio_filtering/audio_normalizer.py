"""Audio normalization and gain control implementation."""


import numpy as np

from .interfaces import AudioNormalizerInterface
from .models import FilterStats


class AudioNormalizer(AudioNormalizerInterface):
    """
    Audio normalization and automatic gain control implementation.

    Provides RMS level detection, automatic gain control with attack/release timing,
    peak limiting, and dynamic range compression for consistent audio levels.
    """

    def __init__(
        self,
        target_level: float = -20.0,
        max_gain: float = 20.0,
        compression_ratio: float = 4.0,
        compression_threshold: float = -12.0,
        attack_time: float = 0.01,  # 10ms
        release_time: float = 0.1,  # 100ms
        sample_rate: int = 16000,
    ):
        """
        Initialize AudioNormalizer.

        Args:
            target_level: Target RMS level in dB
            max_gain: Maximum gain in dB
            compression_ratio: Compression ratio (e.g., 4.0 = 4:1)
            compression_threshold: Compression threshold in dB
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.target_level = target_level
        self.max_gain = max_gain
        self.compression_ratio = compression_ratio
        self.compression_threshold = compression_threshold
        self.attack_time = attack_time
        self.release_time = release_time
        self.sample_rate = sample_rate

        # State variables
        self.current_level = -60.0  # Start with very low level
        self.current_gain = 1.0
        self.envelope_follower = 0.0

        # Calculate attack/release coefficients
        self.attack_coeff = np.exp(-1.0 / (attack_time * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release_time * sample_rate))

        # Peak limiter state
        self.limiter_envelope = 0.0
        self.limiter_threshold = 0.95  # Just below clipping

        # Statistics
        self.processing_latency_ms = 0.0
        self.noise_reduction_db = 0.0
        self.signal_enhancement_db = 0.0

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio levels for optimal processing.

        Applies automatic gain control, dynamic range compression, and peak limiting.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Normalized audio data
        """
        if len(audio_data) == 0:
            return audio_data

        # Apply AGC first
        agc_audio = self.apply_agc(audio_data)

        # Apply dynamic range compression
        compressed_audio = self.compress_dynamic_range(agc_audio)

        # Apply peak limiting as final stage
        limited_audio = self._apply_peak_limiter(compressed_audio)

        return limited_audio

    def apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply automatic gain control to audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with AGC applied
        """
        if len(audio_data) == 0:
            return audio_data

        # Calculate overall RMS level for the chunk
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
        else:
            rms_db = -60.0

        # Update current level
        self.current_level = rms_db

        # Simple AGC: calculate gain needed to reach target level
        gain_needed_db = self.target_level - rms_db
        gain_needed_db = np.clip(gain_needed_db, -self.max_gain, self.max_gain)
        gain_needed = 10 ** (gain_needed_db / 20.0)

        # Apply sample-by-sample processing with envelope following
        output_audio = np.zeros_like(audio_data)

        for i, sample in enumerate(audio_data):
            # Calculate instantaneous level
            instant_level = abs(sample)

            # Update envelope follower
            if instant_level > self.envelope_follower:
                # Attack - fast response to increases
                self.envelope_follower = (
                    self.attack_coeff * self.envelope_follower
                    + (1 - self.attack_coeff) * instant_level
                )
            else:
                # Release - slow response to decreases
                self.envelope_follower = (
                    self.release_coeff * self.envelope_follower
                    + (1 - self.release_coeff) * instant_level
                )

            # Calculate gain based on envelope - this is where AGC compression happens
            if self.envelope_follower > 0:
                # Convert envelope to dB
                envelope_db = 20 * np.log10(self.envelope_follower + 1e-10)

                # Calculate gain reduction needed
                gain_reduction_db = max(0, envelope_db - self.target_level)
                gain_reduction = 10 ** (-gain_reduction_db / 20.0)

                # Combine with overall gain needed
                total_gain = gain_needed * gain_reduction

                # Limit total gain
                total_gain = np.clip(
                    total_gain,
                    10 ** (-self.max_gain / 20.0),
                    10 ** (self.max_gain / 20.0),
                )
            else:
                total_gain = gain_needed

            # Smooth gain changes
            if total_gain < self.current_gain:
                # Gain reduction (attack) - should be fast
                self.current_gain = (
                    self.attack_coeff * self.current_gain
                    + (1 - self.attack_coeff) * total_gain
                )
            else:
                # Gain increase (release) - should be slow
                self.current_gain = (
                    self.release_coeff * self.current_gain
                    + (1 - self.release_coeff) * total_gain
                )

            # Apply gain
            output_audio[i] = sample * self.current_gain

        return output_audio

    def compress_dynamic_range(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Compressed audio data
        """
        if len(audio_data) == 0:
            return audio_data

        # Convert threshold to linear
        threshold_linear = 10 ** (self.compression_threshold / 20.0)

        output_audio = np.zeros_like(audio_data)
        compressor_envelope = 0.0

        for i, sample in enumerate(audio_data):
            # Calculate instantaneous level
            instant_level = abs(sample)

            # Update compressor envelope
            if instant_level > compressor_envelope:
                compressor_envelope = (
                    self.attack_coeff * compressor_envelope
                    + (1 - self.attack_coeff) * instant_level
                )
            else:
                compressor_envelope = (
                    self.release_coeff * compressor_envelope
                    + (1 - self.release_coeff) * instant_level
                )

            # Calculate compression gain
            if compressor_envelope > threshold_linear:
                # Above threshold - apply compression
                over_threshold_db = 20 * np.log10(compressor_envelope / threshold_linear)
                compressed_db = over_threshold_db / self.compression_ratio
                compression_gain = 10 ** ((compressed_db - over_threshold_db) / 20.0)
            else:
                # Below threshold - no compression
                compression_gain = 1.0

            # Apply compression
            output_audio[i] = sample * compression_gain

        return output_audio

    def _apply_peak_limiter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply peak limiting to prevent clipping.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Peak-limited audio data
        """
        if len(audio_data) == 0:
            return audio_data

        output_audio = np.zeros_like(audio_data)

        # Very fast attack for limiter (1 sample)
        limiter_attack_coeff = 0.0
        limiter_release_coeff = np.exp(-1.0 / (0.01 * self.sample_rate))  # 10ms release

        for i, sample in enumerate(audio_data):
            # Calculate instantaneous level
            instant_level = abs(sample)

            # Update limiter envelope
            if instant_level > self.limiter_envelope:
                # Instant attack for limiting
                self.limiter_envelope = instant_level
            else:
                # Slow release
                self.limiter_envelope = (
                    limiter_release_coeff * self.limiter_envelope
                    + (1 - limiter_release_coeff) * instant_level
                )

            # Calculate limiting gain
            if self.limiter_envelope > self.limiter_threshold:
                limiting_gain = self.limiter_threshold / self.limiter_envelope
            else:
                limiting_gain = 1.0

            # Apply limiting
            output_audio[i] = sample * limiting_gain

        return output_audio

    def get_current_level(self) -> float:
        """
        Get the current audio level.

        Returns:
            Current audio level in dB
        """
        return self.current_level

    def get_filter_stats(self) -> FilterStats:
        """
        Get statistics about the filtering performance.

        Returns:
            FilterStats containing performance metrics
        """
        return FilterStats(
            noise_reduction_db=self.noise_reduction_db,
            signal_enhancement_db=self.signal_enhancement_db,
            processing_latency_ms=self.processing_latency_ms,
            filters_applied=["AGC", "Compressor", "Peak Limiter"],
            audio_quality_score=0.8,  # Placeholder
        )

    def reset(self) -> None:
        """Reset the filter state and adaptive parameters."""
        self.current_level = -60.0
        self.current_gain = 1.0
        self.envelope_follower = 0.0
        self.limiter_envelope = 0.0
