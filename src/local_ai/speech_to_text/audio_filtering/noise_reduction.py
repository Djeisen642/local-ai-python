"""Noise reduction engine implementation using spectral subtraction."""

from typing import Optional

import numpy as np
from scipy import signal

from .interfaces import NoiseReductionInterface
from .models import NoiseType


class NoiseReductionEngine(NoiseReductionInterface):
    """Noise reduction engine using spectral subtraction and adaptive filtering.

    Implements FFT-based spectral analysis for noise profiling and basic
    spectral subtraction algorithm for stationary noise removal.
    """

    def __init__(self, sample_rate: int, aggressiveness: float = 0.5) -> None:
        """Initialize the noise reduction engine.

        Args:
            sample_rate: Audio sample rate in Hz
            aggressiveness: Noise reduction aggressiveness (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))

        # Noise profile storage
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_profile_count = 0
        self.adaptation_rate = 0.1

        # Spectral subtraction parameters
        self.alpha = 2.0 + self.aggressiveness * 2.0  # Over-subtraction factor
        self.beta = 0.01 + self.aggressiveness * 0.04  # Spectral floor

        # Wiener filter parameters
        self.use_wiener_filter = True
        self.wiener_alpha = 0.98  # Smoothing factor for Wiener coefficients
        self.speech_presence_threshold = 0.5
        self.previous_wiener_gain: Optional[np.ndarray] = None

        # FFT parameters
        self.fft_size = 512
        self.hop_size = self.fft_size // 4
        self.window = np.hanning(self.fft_size)

        # Noise type detection parameters
        self.stationarity_threshold = 0.8
        self.transient_threshold = 10.0  # Peak-to-average ratio
        self.mechanical_harmonic_threshold = 3

        # Performance tracking
        self.last_noise_reduction_db = 0.0

    def update_noise_profile(self, audio_data: np.ndarray) -> None:
        """Update the noise profile based on audio data.

        Learns noise characteristics during silent periods for adaptive filtering.

        Args:
            audio_data: Audio data to analyze for noise characteristics
        """
        if len(audio_data) < self.fft_size:
            return

        # Compute power spectral density
        freqs, psd = signal.welch(
            audio_data,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.fft_size,
            noverlap=self.fft_size // 2,
        )

        if self.noise_profile is None:
            # Initialize noise profile
            self.noise_profile = psd.copy()
            self.noise_profile_count = 1
        else:
            # Adaptive update using exponential moving average
            self.noise_profile = (
                1 - self.adaptation_rate
            ) * self.noise_profile + self.adaptation_rate * psd
            self.noise_profile_count += 1

    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio data using spectral subtraction.

        Args:
            audio_data: Input audio data as numpy array

        Returns:
            Noise-reduced audio data
        """
        if self.noise_profile is None or len(audio_data) < self.fft_size:
            return audio_data

        # Pad audio to ensure we can process it
        padded_length = len(audio_data) + self.fft_size
        padded_audio = np.pad(
            audio_data, (0, padded_length - len(audio_data)), "constant"
        )

        # Process audio in overlapping frames
        output = np.zeros_like(padded_audio)

        for i in range(0, len(padded_audio) - self.fft_size + 1, self.hop_size):
            frame = padded_audio[i : i + self.fft_size] * self.window

            # Forward FFT
            frame_fft = np.fft.fft(frame)
            frame_magnitude = np.abs(frame_fft)
            frame_phase = np.angle(frame_fft)

            # Apply noise reduction (spectral subtraction + Wiener filtering)
            if self.use_wiener_filter and self.noise_profile is not None:
                enhanced_magnitude = self._adaptive_wiener_filter(frame_magnitude)
            else:
                enhanced_magnitude = self._spectral_subtraction(frame_magnitude)

            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * frame_phase)
            enhanced_frame = np.real(np.fft.ifft(enhanced_fft))

            # Overlap-add
            output[i : i + self.fft_size] += enhanced_frame * self.window

        # Calculate noise reduction achieved
        self._calculate_noise_reduction(audio_data, output[: len(audio_data)])

        return output[: len(audio_data)]

    def _spectral_subtraction(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction algorithm.

        Args:
            magnitude: Magnitude spectrum of the noisy signal

        Returns:
            Enhanced magnitude spectrum
        """
        # Interpolate noise profile to match current frame size
        noise_magnitude = np.interp(
            np.linspace(0, 1, len(magnitude)),
            np.linspace(0, 1, len(self.noise_profile)),
            np.sqrt(self.noise_profile),
        )

        # Spectral subtraction with over-subtraction factor
        enhanced_magnitude = magnitude - self.alpha * noise_magnitude

        # Apply spectral floor to prevent over-subtraction artifacts
        spectral_floor = self.beta * magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)

        return enhanced_magnitude

    def _adaptive_wiener_filter(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply adaptive Wiener filtering for speech preservation.

        Args:
            magnitude: Magnitude spectrum of the noisy signal

        Returns:
            Enhanced magnitude spectrum using Wiener filtering
        """
        # Interpolate noise profile to match current frame size
        noise_magnitude = np.interp(
            np.linspace(0, 1, len(magnitude)),
            np.linspace(0, 1, len(self.noise_profile)),
            np.sqrt(self.noise_profile),
        )

        # Estimate signal power and noise power
        signal_power = magnitude**2
        noise_power = noise_magnitude**2

        # Calculate a priori SNR estimate
        snr_prior = self._estimate_prior_snr(signal_power, noise_power)

        # Calculate speech presence probability
        speech_prob = self._estimate_speech_presence(signal_power, noise_power)

        # Calculate Wiener gain
        wiener_gain = self._calculate_wiener_gain(snr_prior, speech_prob)

        # Apply temporal smoothing to Wiener gain
        if self.previous_wiener_gain is not None and len(
            self.previous_wiener_gain
        ) == len(wiener_gain):
            wiener_gain = (
                self.wiener_alpha * self.previous_wiener_gain
                + (1 - self.wiener_alpha) * wiener_gain
            )

        self.previous_wiener_gain = wiener_gain.copy()

        # Apply Wiener filter
        enhanced_magnitude = wiener_gain * magnitude

        # Apply spectral floor to prevent over-suppression
        spectral_floor = self.beta * magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)

        return enhanced_magnitude

    def _estimate_prior_snr(
        self, signal_power: np.ndarray, noise_power: np.ndarray
    ) -> np.ndarray:
        """Estimate a priori SNR for Wiener filtering.

        Args:
            signal_power: Estimated signal power spectrum
            noise_power: Estimated noise power spectrum

        Returns:
            A priori SNR estimate
        """
        # Simple a priori SNR estimation
        # In practice, this could use more sophisticated methods like decision-directed approach
        snr_prior = (signal_power - noise_power) / (noise_power + 1e-10)

        # Ensure SNR is non-negative and reasonable
        snr_prior = np.maximum(snr_prior, 0.01)  # Minimum SNR
        snr_prior = np.minimum(snr_prior, 100.0)  # Maximum SNR

        return snr_prior

    def _estimate_speech_presence(
        self, signal_power: np.ndarray, noise_power: np.ndarray
    ) -> np.ndarray:
        """Estimate speech presence probability.

        Args:
            signal_power: Signal power spectrum
            noise_power: Noise power spectrum

        Returns:
            Speech presence probability (0 to 1)
        """
        # Calculate instantaneous SNR
        snr_inst = signal_power / (noise_power + 1e-10)

        # Convert to dB
        snr_db = 10 * np.log10(snr_inst + 1e-10)

        # Simple speech presence estimation based on SNR
        # Higher SNR indicates higher probability of speech
        speech_prob = 1.0 / (1.0 + np.exp(-(snr_db - 3.0) / 2.0))  # Sigmoid function

        return speech_prob

    def _calculate_wiener_gain(
        self, snr_prior: np.ndarray, speech_prob: np.ndarray
    ) -> np.ndarray:
        """Calculate Wiener filter gain coefficients.

        Args:
            snr_prior: A priori SNR estimate
            speech_prob: Speech presence probability

        Returns:
            Wiener filter gain coefficients
        """
        # Standard Wiener filter gain
        wiener_gain = snr_prior / (1.0 + snr_prior)

        # Modulate gain based on speech presence probability
        # Preserve more signal when speech is likely present
        modulated_gain = speech_prob * wiener_gain + (1 - speech_prob) * wiener_gain * 0.1

        # Apply aggressiveness factor
        # More aggressive = more noise reduction, less speech preservation
        final_gain = (
            1 - self.aggressiveness
        ) * modulated_gain + self.aggressiveness * wiener_gain

        # Ensure gain is between 0 and 1
        final_gain = np.clip(final_gain, 0.0, 1.0)

        return final_gain

    def detect_noise_type(self, audio_data: np.ndarray) -> NoiseType:
        """Detect the type of noise present in the audio.

        Args:
            audio_data: Audio data to analyze

        Returns:
            Detected noise type
        """
        if len(audio_data) < self.fft_size:
            return NoiseType.MIXED

        # Analyze temporal characteristics
        energy = audio_data**2
        peak_to_avg_ratio = np.max(energy) / (np.mean(energy) + 1e-10)

        # Check for transient characteristics
        if peak_to_avg_ratio > self.transient_threshold:
            return NoiseType.TRANSIENT

        # Analyze spectral characteristics
        freqs, psd = signal.welch(
            audio_data,
            fs=self.sample_rate,
            window="hann",
            nperseg=min(self.fft_size, len(audio_data)),
        )

        # Check for stationarity by analyzing spectral consistency
        stationarity = self._measure_stationarity(audio_data)

        if stationarity > self.stationarity_threshold:
            # Check for harmonic structure (mechanical noise)
            harmonic_count = self._count_harmonics(freqs, psd)

            if harmonic_count >= self.mechanical_harmonic_threshold:
                return NoiseType.MECHANICAL
            else:
                return NoiseType.STATIONARY

        # Check for speech-like characteristics
        if self._detect_speech_characteristics(freqs, psd):
            return NoiseType.SPEECH

        return NoiseType.MIXED

    def _measure_stationarity(self, audio_data: np.ndarray) -> float:
        """Measure the stationarity of the audio signal.

        Args:
            audio_data: Audio data to analyze

        Returns:
            Stationarity measure (0.0 to 1.0)
        """
        if len(audio_data) < 4 * self.fft_size:
            return 0.5  # Default for short signals

        # Split into segments and compare spectral characteristics
        segment_size = len(audio_data) // 4
        segments = [
            audio_data[i : i + segment_size]
            for i in range(0, len(audio_data), segment_size)
        ][:4]

        # Compute PSD for each segment
        psds = []
        for segment in segments:
            if len(segment) >= self.fft_size:
                _, psd = signal.welch(
                    segment,
                    fs=self.sample_rate,
                    window="hann",
                    nperseg=min(self.fft_size, len(segment)),
                )
                psds.append(psd)

        if len(psds) < 2:
            return 0.5

        # Calculate correlation between segments
        correlations = []
        for i in range(1, len(psds)):
            # Ensure same length for correlation
            min_len = min(len(psds[0]), len(psds[i]))
            corr = np.corrcoef(psds[0][:min_len], psds[i][:min_len])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return np.mean(correlations) if correlations else 0.5

    def _count_harmonics(self, freqs: np.ndarray, psd: np.ndarray) -> int:
        """Count harmonic peaks in the power spectral density.

        Args:
            freqs: Frequency array
            psd: Power spectral density

        Returns:
            Number of detected harmonic peaks
        """
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(psd, height=np.max(psd) * 0.1)

        if len(peaks) < 2:
            return 0

        # Look for harmonic relationships
        peak_freqs = freqs[peaks]
        harmonic_count = 0

        # Check for integer multiples (harmonics)
        for i, f1 in enumerate(peak_freqs[:-1]):
            for f2 in peak_freqs[i + 1 :]:
                if f1 > 0 and f2 > f1:
                    ratio = f2 / f1
                    # Check if ratio is close to an integer (harmonic relationship)
                    if abs(ratio - round(ratio)) < 0.1:
                        harmonic_count += 1

        return harmonic_count

    def _detect_speech_characteristics(self, freqs: np.ndarray, psd: np.ndarray) -> bool:
        """Detect if the signal has speech-like characteristics.

        Args:
            freqs: Frequency array
            psd: Power spectral density

        Returns:
            True if speech characteristics are detected
        """
        # Check for energy in typical speech frequency ranges
        speech_bands = [
            (200, 400),  # Fundamental frequency range
            (400, 800),  # First formant range
            (800, 2500),  # Second formant range
            (2500, 4000),  # Higher formants
        ]

        speech_energy = 0
        total_energy = np.sum(psd)

        for low_freq, high_freq in speech_bands:
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                speech_energy += np.sum(psd[band_mask])

        # Speech should have significant energy in speech bands
        speech_ratio = speech_energy / (total_energy + 1e-10)
        return speech_ratio > 0.6

    def _calculate_noise_reduction(
        self, original: np.ndarray, enhanced: np.ndarray
    ) -> None:
        """Calculate the noise reduction achieved in dB.

        Args:
            original: Original noisy audio
            enhanced: Enhanced audio after noise reduction
        """
        if self.noise_profile is None:
            self.last_noise_reduction_db = 0.0
            return

        # Calculate noise reduction based on power difference
        original_power = np.mean(original**2)
        enhanced_power = np.mean(enhanced**2)

        # Simple noise reduction estimate
        if enhanced_power > 0 and original_power > 0:
            # Calculate power ratio
            power_ratio = original_power / enhanced_power
            if power_ratio > 1.0:
                self.last_noise_reduction_db = 10 * np.log10(power_ratio)
            else:
                self.last_noise_reduction_db = 0.0
        else:
            self.last_noise_reduction_db = 0.0

    def get_noise_reduction_db(self) -> float:
        """Get the current noise reduction level in dB.

        Returns:
            Noise reduction level in decibels
        """
        return self.last_noise_reduction_db
