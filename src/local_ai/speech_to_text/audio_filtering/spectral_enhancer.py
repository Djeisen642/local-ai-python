"""SpectralEnhancer implementation for frequency domain audio processing."""

from typing import Optional

import numpy as np
from scipy import signal

from .interfaces import SpectralEnhancerInterface


class SpectralEnhancer(SpectralEnhancerInterface):
    """Frequency domain processing for speech enhancement and noise reduction.

    This class implements various spectral enhancement techniques including:
    - High-pass filtering for low-frequency noise removal
    - Speech band enhancement (300-3400Hz)
    - Echo reduction using frequency domain techniques
    - Transient noise suppression for keyboard clicks and mechanical noise
    """

    def __init__(self, sample_rate: int = 16000):
        """Initialize SpectralEnhancer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0

        # Filter design parameters
        self._high_pass_order = 4
        self._speech_band_low = 300.0
        self._speech_band_high = 3400.0
        self._enhancement_factor = 1.5

        # Echo reduction parameters
        self._echo_frame_size = 1024
        self._echo_hop_size = 512
        self._echo_threshold = 0.3

        # Transient suppression parameters
        self._transient_frame_size = 512
        self._transient_hop_size = 256
        self._transient_threshold = 2.0
        self._spectral_floor = 0.01

        # Pre-compute filter coefficients
        self._precompute_filters()

    def _precompute_filters(self) -> None:
        """Pre-compute filter coefficients for efficiency."""
        # High-pass filter coefficients (will be computed per cutoff)
        self._hp_filters = {}

        # Speech enhancement filter bank
        self._compute_speech_enhancement_filters()

    def _compute_speech_enhancement_filters(self) -> None:
        """Compute filter bank for speech enhancement."""
        # Create frequency bins for enhancement
        fft_size = 1024
        freqs = np.fft.fftfreq(fft_size, 1 / self.sample_rate)[: fft_size // 2 + 1]

        # Speech band enhancement curve
        self.speech_enhancement_curve = np.ones_like(freqs)

        # Enhance speech band (300-3400Hz)
        speech_mask = (freqs >= self._speech_band_low) & (freqs <= self._speech_band_high)
        self.speech_enhancement_curve[speech_mask] = self._enhancement_factor

        # Gentle roll-off outside speech band
        below_speech = freqs < self._speech_band_low
        above_speech = freqs > self._speech_band_high

        # Gradual transition zones
        transition_width = 200.0  # Hz

        # Below speech band - gradual reduction
        for i, freq in enumerate(freqs):
            if freq < self._speech_band_low:
                if freq > self._speech_band_low - transition_width:
                    # Linear transition
                    factor = (
                        freq - (self._speech_band_low - transition_width)
                    ) / transition_width
                    self.speech_enhancement_curve[i] = 1.0 + factor * (
                        self._enhancement_factor - 1.0
                    )
                else:
                    self.speech_enhancement_curve[i] = 1.0

        # Above speech band - gradual reduction
        for i, freq in enumerate(freqs):
            if freq > self._speech_band_high:
                if freq < self._speech_band_high + transition_width:
                    # Linear transition
                    factor = 1.0 - (freq - self._speech_band_high) / transition_width
                    self.speech_enhancement_curve[i] = 1.0 + factor * (
                        self._enhancement_factor - 1.0
                    )
                else:
                    self.speech_enhancement_curve[i] = 1.0

    def enhance_speech_frequencies(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance speech frequency bands for better clarity.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with enhanced speech frequencies
        """
        if len(audio_data) == 0:
            return audio_data.copy()

        # Use overlap-add processing for frequency domain enhancement
        frame_size = 1024
        hop_size = 512

        # Pad audio for processing
        padded_length = len(audio_data) + frame_size
        padded_audio = np.zeros(padded_length, dtype=audio_data.dtype)
        padded_audio[: len(audio_data)] = audio_data

        # Output buffer
        enhanced = np.zeros_like(padded_audio)

        # Window function
        window = np.hanning(frame_size)

        # Process in overlapping frames
        for start in range(0, len(padded_audio) - frame_size + 1, hop_size):
            # Extract frame
            frame = padded_audio[start : start + frame_size] * window

            # FFT
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame)
            phase = np.angle(fft_frame)

            # Apply speech enhancement to positive frequencies
            enhanced_magnitude = magnitude.copy()
            n_bins = len(self.speech_enhancement_curve)
            enhanced_magnitude[:n_bins] *= self.speech_enhancement_curve

            # Maintain symmetry for real signal
            if len(enhanced_magnitude) > n_bins:
                enhanced_magnitude[n_bins:] = enhanced_magnitude[
                    1 : len(enhanced_magnitude) - n_bins + 1
                ][::-1]

            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_frame = np.real(np.fft.ifft(enhanced_fft))

            # Apply window and overlap-add
            enhanced_frame *= window
            enhanced[start : start + frame_size] += enhanced_frame

        # Return original length
        return enhanced[: len(audio_data)]

    def apply_high_pass_filter(
        self, audio_data: np.ndarray, cutoff: float = 80.0
    ) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise.

        Args:
            audio_data: Input audio data as numpy array
            cutoff: High-pass filter cutoff frequency in Hz

        Returns:
            High-pass filtered audio data
        """
        if len(audio_data) == 0:
            return audio_data.copy()

        # Ensure cutoff is valid
        cutoff = max(1.0, min(cutoff, self.nyquist * 0.9))

        # Check if filter is already computed
        filter_key = f"hp_{cutoff}_{self._high_pass_order}"
        if filter_key not in self._hp_filters:
            # Design Butterworth high-pass filter
            normalized_cutoff = cutoff / self.nyquist
            self._hp_filters[filter_key] = signal.butter(
                self._high_pass_order, normalized_cutoff, btype="high", output="sos"
            )

        # Apply filter
        sos = self._hp_filters[filter_key]
        filtered = signal.sosfilt(sos, audio_data)

        return filtered.astype(audio_data.dtype)

    def reduce_echo(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce echo and reverberation in audio data.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with reduced echo
        """
        if len(audio_data) == 0:
            return audio_data.copy()

        # Use spectral subtraction approach for echo reduction
        frame_size = self._echo_frame_size
        hop_size = self._echo_hop_size

        # Pad audio
        padded_length = len(audio_data) + frame_size
        padded_audio = np.zeros(padded_length, dtype=audio_data.dtype)
        padded_audio[: len(audio_data)] = audio_data

        # Output buffer
        processed = np.zeros_like(padded_audio)

        # Window function
        window = np.hanning(frame_size)

        # Estimate noise/echo profile from initial frames
        echo_profile = self._estimate_echo_profile(
            padded_audio, frame_size, hop_size, window
        )

        # Process frames
        for start in range(0, len(padded_audio) - frame_size + 1, hop_size):
            frame = padded_audio[start : start + frame_size] * window

            # FFT
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame)
            phase = np.angle(fft_frame)

            # Apply spectral subtraction for echo reduction
            enhanced_magnitude = self._apply_echo_spectral_subtraction(
                magnitude, echo_profile
            )

            # Reconstruct
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_frame = np.real(np.fft.ifft(enhanced_fft))

            # Overlap-add
            enhanced_frame *= window
            processed[start : start + frame_size] += enhanced_frame

        return processed[: len(audio_data)]

    def _estimate_echo_profile(
        self, audio_data: np.ndarray, frame_size: int, hop_size: int, window: np.ndarray
    ) -> np.ndarray:
        """Estimate echo/reverberation profile from audio."""
        # Use first few frames to estimate background echo characteristics
        num_profile_frames = min(5, (len(audio_data) - frame_size) // hop_size + 1)

        magnitude_sum = None
        frame_count = 0

        for i in range(num_profile_frames):
            start = i * hop_size
            if start + frame_size > len(audio_data):
                break

            frame = audio_data[start : start + frame_size] * window
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame)

            if magnitude_sum is None:
                magnitude_sum = magnitude.copy()
            else:
                magnitude_sum += magnitude
            frame_count += 1

        if frame_count > 0:
            echo_profile = magnitude_sum / frame_count
        else:
            echo_profile = np.ones(frame_size)

        return echo_profile

    def _apply_echo_spectral_subtraction(
        self, magnitude: np.ndarray, echo_profile: np.ndarray
    ) -> np.ndarray:
        """Apply spectral subtraction for echo reduction."""
        # More conservative approach for echo reduction
        alpha = 0.5  # Reduced over-subtraction factor
        beta = 0.3  # Higher spectral floor to preserve signal

        # Normalize echo profile to prevent over-subtraction
        if np.max(echo_profile) > 0:
            normalized_echo_profile = (
                echo_profile / np.max(echo_profile) * np.mean(magnitude)
            )
        else:
            normalized_echo_profile = echo_profile

        # Estimate echo component (more conservative)
        echo_estimate = alpha * normalized_echo_profile

        # Subtract echo estimate
        enhanced_magnitude = magnitude - echo_estimate

        # Apply higher spectral floor to preserve signal quality
        spectral_floor = beta * magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)

        return enhanced_magnitude

    def suppress_transients(self, audio_data: np.ndarray) -> np.ndarray:
        """Suppress transient noises like keyboard clicks and pops.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Audio data with suppressed transients
        """
        if len(audio_data) == 0:
            return audio_data.copy()

        # Use time-frequency analysis for transient detection and suppression
        frame_size = self._transient_frame_size
        hop_size = self._transient_hop_size

        # Pad audio
        padded_length = len(audio_data) + frame_size
        padded_audio = np.zeros(padded_length, dtype=audio_data.dtype)
        padded_audio[: len(audio_data)] = audio_data

        # Output buffer
        processed = np.zeros_like(padded_audio)

        # Window function
        window = np.hanning(frame_size)

        # Analyze frames for transient detection
        transient_mask = self._detect_transients(
            padded_audio, frame_size, hop_size, window
        )

        # Process frames with adaptive suppression
        for i, start in enumerate(range(0, len(padded_audio) - frame_size + 1, hop_size)):
            frame = padded_audio[start : start + frame_size] * window

            # Apply suppression based on transient detection
            if i < len(transient_mask) and transient_mask[i]:
                # Strong suppression for detected transients
                suppressed_frame = self._apply_transient_suppression(frame, strong=True)
            else:
                # Light processing for normal audio
                suppressed_frame = self._apply_transient_suppression(frame, strong=False)

            # Overlap-add
            suppressed_frame *= window
            processed[start : start + frame_size] += suppressed_frame

        return processed[: len(audio_data)]

    def _detect_transients(
        self, audio_data: np.ndarray, frame_size: int, hop_size: int, window: np.ndarray
    ) -> np.ndarray:
        """Detect transient events in audio."""
        num_frames = (len(audio_data) - frame_size) // hop_size + 1
        transient_mask = np.zeros(num_frames, dtype=bool)

        # Calculate energy and spectral features for each frame
        energies = []
        spectral_centroids = []

        for i in range(num_frames):
            start = i * hop_size
            if start + frame_size > len(audio_data):
                break

            frame = audio_data[start : start + frame_size] * window

            # Energy calculation
            energy = np.sum(frame**2)
            energies.append(energy)

            # Spectral centroid calculation
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame[: frame_size // 2 + 1])
            freqs = np.fft.fftfreq(frame_size, 1 / self.sample_rate)[
                : frame_size // 2 + 1
            ]

            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0
            spectral_centroids.append(centroid)

        if len(energies) == 0:
            return transient_mask

        # Convert to numpy arrays
        energies = np.array(energies)
        spectral_centroids = np.array(spectral_centroids)

        # Detect transients based on energy and spectral changes
        if len(energies) > 1:
            # Energy-based detection
            energy_diff = np.diff(energies)
            energy_threshold = np.std(energy_diff) * self._transient_threshold

            # Spectral centroid-based detection
            centroid_diff = np.diff(spectral_centroids)
            centroid_threshold = np.std(centroid_diff) * self._transient_threshold

            # Mark frames with sudden changes as transients
            for i in range(1, len(energies)):
                if (
                    abs(energy_diff[i - 1]) > energy_threshold
                    or abs(centroid_diff[i - 1]) > centroid_threshold
                ):
                    transient_mask[i] = True

                    # Also mark neighboring frames
                    if i > 0:
                        transient_mask[i - 1] = True
                    if i < len(transient_mask) - 1:
                        transient_mask[i + 1] = True

        return transient_mask

    def _apply_transient_suppression(
        self, frame: np.ndarray, strong: bool = False
    ) -> np.ndarray:
        """Apply transient suppression to a frame."""
        if strong:
            # More conservative suppression for detected transients
            # Use spectral gating with speech preservation
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame)
            phase = np.angle(fft_frame)

            # Create frequency-dependent suppression
            freqs = np.fft.fftfreq(len(frame), 1 / self.sample_rate)
            suppression_factor = np.ones_like(magnitude)

            # Preserve speech frequencies (300-3400Hz) more
            speech_mask = (np.abs(freqs) >= 300) & (np.abs(freqs) <= 3400)
            transient_mask = (
                np.abs(freqs) > 2000
            )  # High frequencies more likely transients

            # Conservative suppression in speech band
            suppression_factor[speech_mask] = 0.7
            # Stronger suppression in high frequencies
            suppression_factor[transient_mask] = 0.4

            # Apply spectral floor to prevent over-suppression
            spectral_floor = 0.2 * magnitude  # Higher floor
            suppressed_magnitude = np.maximum(
                magnitude * suppression_factor, spectral_floor
            )

            # Reconstruct
            suppressed_fft = suppressed_magnitude * np.exp(1j * phase)
            suppressed_frame = np.real(np.fft.ifft(suppressed_fft))

            return suppressed_frame
        else:
            # Light processing for normal audio
            # Apply gentle high-frequency roll-off to reduce click artifacts
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame)
            phase = np.angle(fft_frame)

            # Gentle high-frequency attenuation
            freqs = np.fft.fftfreq(len(frame), 1 / self.sample_rate)
            hf_rolloff = np.ones_like(magnitude)

            # Attenuate frequencies above 4kHz slightly
            hf_mask = np.abs(freqs) > 4000
            hf_rolloff[hf_mask] = 0.95  # Less aggressive

            processed_magnitude = magnitude * hf_rolloff

            # Reconstruct
            processed_fft = processed_magnitude * np.exp(1j * phase)
            processed_frame = np.real(np.fft.ifft(processed_fft))

            return processed_frame
