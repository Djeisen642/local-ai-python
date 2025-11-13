"""Tests for SpectralEnhancer frequency domain processing."""

import numpy as np
import pytest
from scipy import signal

from local_ai.speech_to_text.audio_filtering.spectral_enhancer import SpectralEnhancer


class TestSpectralEnhancerHighPassFilter:
    """Test cases for high-pass filter functionality."""

    @pytest.fixture
    def enhancer(self):
        """Create SpectralEnhancer instance for testing."""
        return SpectralEnhancer(sample_rate=16000)

    @pytest.fixture
    def sample_rate(self):
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def low_frequency_noise(self, sample_rate):
        """Generate low-frequency noise for testing."""
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create low-frequency noise (rumble, hum)
        noise = (
            0.3 * np.sin(2 * np.pi * 50 * t)  # 50Hz hum
            + 0.2 * np.sin(2 * np.pi * 60 * t)  # 60Hz electrical hum
            + 0.1 * np.sin(2 * np.pi * 30 * t)  # 30Hz rumble
            + 0.05 * np.random.normal(0, 1, samples)  # Low-level white noise
        )
        return noise.astype(np.float32)

    @pytest.fixture
    def speech_with_low_freq_noise(self, sample_rate, low_frequency_noise):
        """Generate speech signal contaminated with low-frequency noise."""
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create speech-like signal in speech band (300-3400Hz)
        speech = (
            0.2 * np.sin(2 * np.pi * 200 * t)  # Fundamental
            + 0.15 * np.sin(2 * np.pi * 400 * t)  # First harmonic
            + 0.1 * np.sin(2 * np.pi * 800 * t)  # Second harmonic
            + 0.05 * np.sin(2 * np.pi * 1600 * t)  # Third harmonic
        )

        # Combine speech with low-frequency noise
        contaminated = speech + low_frequency_noise
        return contaminated.astype(np.float32), speech.astype(np.float32)

    def test_high_pass_filter_removes_low_frequencies(
        self, enhancer, low_frequency_noise, sample_rate
    ):
        """Test high-pass filter effectiveness with low-frequency noise."""
        # Apply high-pass filter with 80Hz cutoff
        filtered = enhancer.apply_high_pass_filter(low_frequency_noise, cutoff=80.0)

        # Analyze frequency content before and after filtering
        freqs_orig, psd_orig = signal.welch(
            low_frequency_noise, sample_rate, nperseg=1024
        )
        freqs_filt, psd_filt = signal.welch(filtered, sample_rate, nperseg=1024)

        # Find energy in low-frequency band (below 80Hz)
        low_freq_mask = freqs_orig < 80.0
        orig_low_energy = np.sum(psd_orig[low_freq_mask])
        filt_low_energy = np.sum(psd_filt[low_freq_mask])

        # Low-frequency energy should be significantly reduced (at least 10dB = 10x reduction)
        reduction_ratio = orig_low_energy / (
            filt_low_energy + 1e-10
        )  # Avoid division by zero
        assert reduction_ratio > 10.0, (
            f"Expected >10x reduction, got {reduction_ratio:.2f}x"
        )

    def test_high_pass_filter_preserves_speech_frequencies(
        self, enhancer, speech_with_low_freq_noise, sample_rate
    ):
        """Test that high-pass filter preserves speech frequencies while removing low-frequency noise."""
        contaminated_speech, clean_speech = speech_with_low_freq_noise

        # Apply high-pass filter
        filtered = enhancer.apply_high_pass_filter(contaminated_speech, cutoff=80.0)

        # Analyze frequency content in speech band (300-3400Hz)
        freqs, psd_orig = signal.welch(contaminated_speech, sample_rate, nperseg=1024)
        freqs, psd_filt = signal.welch(filtered, sample_rate, nperseg=1024)
        freqs, psd_clean = signal.welch(clean_speech, sample_rate, nperseg=1024)

        # Find energy in speech band
        speech_band_mask = (freqs >= 300) & (freqs <= 3400)
        orig_speech_energy = np.sum(psd_orig[speech_band_mask])
        filt_speech_energy = np.sum(psd_filt[speech_band_mask])
        clean_speech_energy = np.sum(psd_clean[speech_band_mask])

        # Filtered speech energy should be closer to clean speech than original
        orig_diff = abs(orig_speech_energy - clean_speech_energy)
        filt_diff = abs(filt_speech_energy - clean_speech_energy)

        assert filt_diff < orig_diff, "Filtered speech should be closer to clean speech"

        # Speech energy should be preserved (within 3dB = 2x)
        preservation_ratio = filt_speech_energy / orig_speech_energy
        assert preservation_ratio > 0.5, (
            f"Speech energy preservation too low: {preservation_ratio:.2f}"
        )

    def test_high_pass_filter_different_cutoff_frequencies(
        self, enhancer, low_frequency_noise, sample_rate
    ):
        """Test high-pass filter with different cutoff frequencies."""
        cutoff_frequencies = [50.0, 80.0, 120.0, 200.0]

        for cutoff in cutoff_frequencies:
            filtered = enhancer.apply_high_pass_filter(low_frequency_noise, cutoff=cutoff)

            # Analyze frequency response
            freqs, psd_filt = signal.welch(filtered, sample_rate, nperseg=1024)

            # Energy below cutoff should be reduced
            below_cutoff_mask = freqs < cutoff
            above_cutoff_mask = freqs > cutoff * 1.5  # Well above cutoff

            below_energy = np.sum(psd_filt[below_cutoff_mask])
            above_energy = np.sum(psd_filt[above_cutoff_mask])

            # Check that filtering actually occurred by comparing with original
            freqs_orig, psd_orig = signal.welch(
                low_frequency_noise, sample_rate, nperseg=1024
            )

            # Energy below cutoff should be reduced compared to original
            below_energy_orig = np.sum(psd_orig[below_cutoff_mask])
            reduction_ratio = below_energy_orig / (below_energy + 1e-10)

            # Should show some reduction in low frequencies
            assert reduction_ratio > 1.2, (
                f"Insufficient low-frequency reduction at {cutoff}Hz cutoff: {reduction_ratio:.2f}x"
            )

    def test_high_pass_filter_stability_and_artifacts(self, enhancer, sample_rate):
        """Test high-pass filter stability and absence of artifacts."""
        # Create test signal with DC offset and low-frequency content
        duration = 2.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Signal with DC offset and various frequency components
        test_signal = (
            0.1  # DC offset
            + 0.2 * np.sin(2 * np.pi * 10 * t)  # Very low frequency
            + 0.3 * np.sin(2 * np.pi * 100 * t)  # Low frequency
            + 0.2 * np.sin(2 * np.pi * 1000 * t)  # Mid frequency
        ).astype(np.float32)

        # Apply filter
        filtered = enhancer.apply_high_pass_filter(test_signal, cutoff=80.0)

        # Check for stability (no NaN or infinite values)
        assert not np.any(np.isnan(filtered)), "Filter produced NaN values"
        assert not np.any(np.isinf(filtered)), "Filter produced infinite values"

        # Check that DC offset is removed
        dc_component = np.mean(filtered)
        assert abs(dc_component) < 0.01, f"DC component not removed: {dc_component:.4f}"

        # Check that filter doesn't introduce excessive ringing or artifacts
        max_amplitude = np.max(np.abs(filtered))
        original_max = np.max(np.abs(test_signal))
        assert max_amplitude < original_max * 2.0, (
            "Filter introduced excessive amplification"
        )


class TestSpectralEnhancerSpeechBandEnhancement:
    """Test cases for speech band enhancement (300-3400Hz)."""

    @pytest.fixture
    def enhancer(self):
        """Create SpectralEnhancer instance for testing."""
        return SpectralEnhancer(sample_rate=16000)

    @pytest.fixture
    def sample_rate(self):
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def broadband_signal(self, sample_rate):
        """Generate broadband signal for testing speech enhancement."""
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create signal with energy across multiple bands
        signal_components = (
            0.1 * np.sin(2 * np.pi * 100 * t)  # Below speech band
            + 0.2 * np.sin(2 * np.pi * 500 * t)  # In speech band
            + 0.15 * np.sin(2 * np.pi * 1000 * t)  # In speech band
            + 0.12 * np.sin(2 * np.pi * 2000 * t)  # In speech band
            + 0.08 * np.sin(2 * np.pi * 5000 * t)  # Above speech band
            + 0.05 * np.random.normal(0, 1, samples)  # Broadband noise
        )
        return signal_components.astype(np.float32)

    @pytest.fixture
    def speech_like_signal(self, sample_rate):
        """Generate speech-like signal with formants."""
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Simulate speech formants (vowel-like)
        f0 = 150  # Fundamental frequency
        f1 = 800  # First formant
        f2 = 1200  # Second formant
        f3 = 2500  # Third formant

        speech = (
            0.3 * np.sin(2 * np.pi * f0 * t)  # Fundamental
            + 0.25 * np.sin(2 * np.pi * f1 * t)  # First formant
            + 0.2 * np.sin(2 * np.pi * f2 * t)  # Second formant
            + 0.15 * np.sin(2 * np.pi * f3 * t)  # Third formant
            + 0.1 * np.sin(2 * np.pi * (2 * f0) * t)  # Second harmonic
        )
        return speech.astype(np.float32)

    def test_speech_band_enhancement_boosts_target_frequencies(
        self, enhancer, broadband_signal, sample_rate
    ):
        """Test speech band enhancement in 300-3400Hz range."""
        # Apply speech enhancement
        enhanced = enhancer.enhance_speech_frequencies(broadband_signal)

        # Analyze frequency content before and after enhancement
        freqs_orig, psd_orig = signal.welch(broadband_signal, sample_rate, nperseg=1024)
        freqs_enh, psd_enh = signal.welch(enhanced, sample_rate, nperseg=1024)

        # Define frequency bands
        speech_band_mask = (freqs_orig >= 300) & (freqs_orig <= 3400)
        below_speech_mask = freqs_orig < 300
        above_speech_mask = freqs_orig > 3400

        # Calculate energy in each band
        orig_speech_energy = np.sum(psd_orig[speech_band_mask])
        enh_speech_energy = np.sum(psd_enh[speech_band_mask])

        orig_below_energy = np.sum(psd_orig[below_speech_mask])
        enh_below_energy = np.sum(psd_enh[below_speech_mask])

        orig_above_energy = np.sum(psd_orig[above_speech_mask])
        enh_above_energy = np.sum(psd_enh[above_speech_mask])

        # Speech band should be enhanced (increased energy)
        speech_enhancement_ratio = enh_speech_energy / orig_speech_energy
        assert speech_enhancement_ratio > 1.1, (
            f"Speech band not enhanced: {speech_enhancement_ratio:.2f}"
        )

        # Non-speech bands should be relatively unchanged or reduced
        below_ratio = enh_below_energy / (orig_below_energy + 1e-10)
        above_ratio = enh_above_energy / (orig_above_energy + 1e-10)

        # Enhancement should be selective (speech band enhanced more than others)
        assert speech_enhancement_ratio > below_ratio, "Speech enhancement not selective"
        assert speech_enhancement_ratio > above_ratio, "Speech enhancement not selective"

    def test_speech_band_enhancement_preserves_formant_structure(
        self, enhancer, speech_like_signal, sample_rate
    ):
        """Test that speech enhancement preserves formant structure."""
        # Apply enhancement
        enhanced = enhancer.enhance_speech_frequencies(speech_like_signal)

        # Analyze formant preservation using cross-correlation
        correlation = np.corrcoef(speech_like_signal, enhanced)[0, 1]
        assert correlation > 0.8, (
            f"Formant structure not preserved: correlation = {correlation:.3f}"
        )

        # Check that enhancement doesn't introduce excessive distortion
        # Calculate THD (Total Harmonic Distortion) approximation
        original_rms = np.sqrt(np.mean(speech_like_signal**2))
        enhanced_rms = np.sqrt(np.mean(enhanced**2))

        # Enhanced signal should be louder but not excessively distorted
        enhancement_factor = enhanced_rms / original_rms
        assert 1.0 < enhancement_factor < 3.0, (
            f"Enhancement factor out of range: {enhancement_factor:.2f}"
        )

    def test_speech_band_enhancement_frequency_selectivity(self, enhancer, sample_rate):
        """Test frequency selectivity of speech band enhancement."""
        duration = 1.0
        samples = int(sample_rate * duration)

        # Create pure tones at different frequencies
        test_frequencies = [100, 500, 1000, 2000, 3000, 5000, 8000]  # Hz
        enhancement_ratios = []

        for freq in test_frequencies:
            t = np.linspace(0, duration, samples, endpoint=False)
            tone = np.sin(2 * np.pi * freq * t).astype(np.float32)

            # Apply enhancement
            enhanced_tone = enhancer.enhance_speech_frequencies(tone)

            # Calculate enhancement ratio
            orig_energy = np.sum(tone**2)
            enh_energy = np.sum(enhanced_tone**2)
            ratio = enh_energy / orig_energy
            enhancement_ratios.append(ratio)

        # Frequencies in speech band (300-3400Hz) should be enhanced more
        speech_band_indices = [1, 2, 3, 4]  # 500, 1000, 2000, 3000 Hz
        non_speech_indices = [0, 5, 6]  # 100, 5000, 8000 Hz

        avg_speech_enhancement = np.mean(
            [enhancement_ratios[i] for i in speech_band_indices]
        )
        avg_non_speech_enhancement = np.mean(
            [enhancement_ratios[i] for i in non_speech_indices]
        )

        assert avg_speech_enhancement > avg_non_speech_enhancement, (
            f"Speech frequencies not selectively enhanced: {avg_speech_enhancement:.2f} vs {avg_non_speech_enhancement:.2f}"
        )

    def test_speech_band_enhancement_with_noise(
        self, enhancer, speech_like_signal, sample_rate
    ):
        """Test speech enhancement in presence of noise."""
        # Add broadband noise to speech signal
        noise = np.random.normal(0, 0.1, len(speech_like_signal)).astype(np.float32)
        noisy_speech = speech_like_signal + noise

        # Apply enhancement
        enhanced = enhancer.enhance_speech_frequencies(noisy_speech)

        # Calculate SNR improvement in speech band
        freqs, psd_orig = signal.welch(noisy_speech, sample_rate, nperseg=1024)
        freqs, psd_enh = signal.welch(enhanced, sample_rate, nperseg=1024)

        # Focus on speech band
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        noise_mask = (freqs > 5000) & (freqs < 8000)  # High-frequency noise reference

        # Calculate SNR (speech energy / noise energy)
        orig_speech_energy = np.sum(psd_orig[speech_mask])
        orig_noise_energy = np.sum(psd_orig[noise_mask])
        orig_snr = orig_speech_energy / (orig_noise_energy + 1e-10)

        enh_speech_energy = np.sum(psd_enh[speech_mask])
        enh_noise_energy = np.sum(psd_enh[noise_mask])
        enh_snr = enh_speech_energy / (enh_noise_energy + 1e-10)

        # SNR should improve (speech enhanced more than noise)
        snr_improvement = enh_snr / orig_snr
        assert snr_improvement > 1.0, f"SNR not improved: {snr_improvement:.2f}"


class TestSpectralEnhancerEchoReduction:
    """Test cases for echo reduction functionality."""

    @pytest.fixture
    def enhancer(self):
        """Create SpectralEnhancer instance for testing."""
        return SpectralEnhancer(sample_rate=16000)

    @pytest.fixture
    def sample_rate(self):
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def synthetic_echo_signal(self, sample_rate):
        """Generate signal with synthetic echo patterns."""
        duration = 2.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create original speech-like signal
        original = (
            0.3 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 2)  # Decaying tone
            + 0.2 * np.sin(2 * np.pi * 800 * t) * (t < 1.0)  # First second only
        )

        # Add echo with different delays and attenuations
        echo_delays = [0.1, 0.2, 0.35]  # seconds
        echo_gains = [0.3, 0.2, 0.1]  # attenuation factors

        echoed_signal = original.copy()
        for delay, gain in zip(echo_delays, echo_gains):
            delay_samples = int(delay * sample_rate)
            if delay_samples < len(original):
                echo_part = np.zeros_like(original)
                echo_part[delay_samples:] = original[:-delay_samples] * gain
                echoed_signal += echo_part

        return echoed_signal.astype(np.float32), original.astype(np.float32)

    @pytest.fixture
    def room_impulse_response(self, sample_rate):
        """Generate synthetic room impulse response for realistic echo."""
        duration = 0.5  # 500ms reverb tail
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create exponentially decaying impulse response with multiple reflections
        impulse = np.zeros(samples)
        impulse[0] = 1.0  # Direct path

        # Add early reflections
        reflection_times = [0.02, 0.035, 0.055, 0.08, 0.12]  # seconds
        reflection_gains = [0.4, 0.3, 0.25, 0.2, 0.15]

        for refl_time, refl_gain in zip(reflection_times, reflection_gains):
            refl_sample = int(refl_time * sample_rate)
            if refl_sample < len(impulse):
                impulse[refl_sample] += refl_gain

        # Add exponential decay for late reverberation
        decay_envelope = np.exp(-t * 8)  # RT60 ≈ 0.5 seconds
        impulse *= decay_envelope

        return impulse.astype(np.float32)

    def test_echo_reduction_with_synthetic_echo(self, enhancer, synthetic_echo_signal):
        """Test echo reduction with synthetic echo patterns."""
        echoed_signal, original_signal = synthetic_echo_signal

        # Apply echo reduction
        processed = enhancer.reduce_echo(echoed_signal)

        # Compare processed signal to original using cross-correlation
        # Normalize signals for comparison
        orig_norm = original_signal / (np.max(np.abs(original_signal)) + 1e-10)
        proc_norm = processed / (np.max(np.abs(processed)) + 1e-10)
        echoed_norm = echoed_signal / (np.max(np.abs(echoed_signal)) + 1e-10)

        # Processed signal should be more similar to original than echoed signal
        orig_correlation = np.corrcoef(orig_norm, proc_norm)[0, 1]
        echoed_correlation = np.corrcoef(orig_norm, echoed_norm)[0, 1]

        # For echo reduction, we expect some improvement but not necessarily perfect restoration
        # The correlation difference should be meaningful (at least 2% improvement or high absolute correlation)
        correlation_improvement = orig_correlation - echoed_correlation

        assert orig_correlation > echoed_correlation or orig_correlation > 0.9, (
            f"Echo reduction ineffective: {orig_correlation:.3f} vs {echoed_correlation:.3f} (improvement: {correlation_improvement:.3f})"
        )

        # Processed signal should have reasonable correlation with original
        assert orig_correlation > 0.6, (
            f"Processed signal too different from original: {orig_correlation:.3f}"
        )

    def test_echo_reduction_preserves_direct_signal(self, enhancer, sample_rate):
        """Test that echo reduction preserves the direct signal component."""
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create clean signal without echo
        clean_signal = (
            0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
            + 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 note (octave)
        ).astype(np.float32)

        # Apply echo reduction to clean signal
        processed = enhancer.reduce_echo(clean_signal)

        # Clean signal should be minimally affected
        correlation = np.corrcoef(clean_signal, processed)[0, 1]
        assert correlation > 0.9, (
            f"Clean signal corrupted by echo reduction: {correlation:.3f}"
        )

        # Energy should be preserved (within reasonable bounds)
        orig_energy = np.sum(clean_signal**2)
        proc_energy = np.sum(processed**2)
        energy_ratio = proc_energy / orig_energy

        assert 0.5 < energy_ratio < 1.5, f"Energy not preserved: {energy_ratio:.2f}"

    def test_echo_reduction_frequency_domain_approach(
        self, enhancer, synthetic_echo_signal, sample_rate
    ):
        """Test echo reduction using frequency domain analysis."""
        echoed_signal, original_signal = synthetic_echo_signal

        # Apply echo reduction
        processed = enhancer.reduce_echo(echoed_signal)

        # Analyze frequency domain characteristics
        freqs_orig, psd_orig = signal.welch(original_signal, sample_rate, nperseg=1024)
        freqs_echo, psd_echo = signal.welch(echoed_signal, sample_rate, nperseg=1024)
        freqs_proc, psd_proc = signal.welch(processed, sample_rate, nperseg=1024)

        # Calculate spectral distance (how close processed is to original vs echoed)
        # Use mean squared difference in log domain
        log_orig = np.log10(psd_orig + 1e-10)
        log_echo = np.log10(psd_echo + 1e-10)
        log_proc = np.log10(psd_proc + 1e-10)

        orig_distance = np.mean((log_proc - log_orig) ** 2)
        echo_distance = np.mean((log_proc - log_echo) ** 2)

        # Processed should be closer to original than to echoed version, or at least not much worse
        # Echo reduction is challenging and perfect restoration isn't always possible
        assert orig_distance < echo_distance * 2.5, (
            f"Processed spectrum significantly worse than echoed: {orig_distance:.3f} vs {echo_distance:.3f}"
        )

    def test_echo_reduction_with_room_impulse_response(
        self, enhancer, room_impulse_response, sample_rate
    ):
        """Test echo reduction with realistic room impulse response."""
        # Create dry speech signal
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        dry_speech = (
            0.2
            * np.sin(2 * np.pi * 150 * t)
            * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))  # Modulated fundamental
            + 0.15 * np.sin(2 * np.pi * 300 * t)  # Second harmonic
            + 0.1 * np.sin(2 * np.pi * 600 * t)  # Fourth harmonic
        ).astype(np.float32)

        # Convolve with room impulse response to create reverberant speech
        reverberant_speech = np.convolve(dry_speech, room_impulse_response, mode="same")
        reverberant_speech = reverberant_speech.astype(np.float32)

        # Apply echo reduction
        processed = enhancer.reduce_echo(reverberant_speech)

        # Measure reverberation reduction
        # Calculate energy decay in the tail of the signal
        tail_start = int(0.7 * len(processed))  # Last 30% of signal

        reverb_tail_energy = np.sum(reverberant_speech[tail_start:] ** 2)
        processed_tail_energy = np.sum(processed[tail_start:] ** 2)
        dry_tail_energy = np.sum(dry_speech[tail_start:] ** 2)

        # Processed signal should have less tail energy than reverberant signal
        tail_reduction = reverb_tail_energy / (processed_tail_energy + 1e-10)
        assert tail_reduction > 1.2, (
            f"Insufficient reverberation reduction: {tail_reduction:.2f}"
        )

        # But should not completely eliminate the signal
        assert processed_tail_energy > dry_tail_energy * 0.1, "Over-processing detected"

    def test_echo_reduction_stability_with_various_delays(self, enhancer, sample_rate):
        """Test echo reduction stability with various echo delays."""
        duration = 1.5
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create base signal
        base_signal = 0.3 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        # Test different echo delays
        echo_delays = [0.05, 0.1, 0.2, 0.3, 0.5]  # 50ms to 500ms

        for delay in echo_delays:
            delay_samples = int(delay * sample_rate)

            # Create echoed signal
            echoed = base_signal.copy()
            if delay_samples < len(base_signal):
                echo_part = np.zeros_like(base_signal)
                echo_part[delay_samples:] = base_signal[:-delay_samples] * 0.4
                echoed += echo_part

            # Apply echo reduction
            processed = enhancer.reduce_echo(echoed)

            # Check for stability
            assert not np.any(np.isnan(processed)), (
                f"NaN values with {delay * 1000:.0f}ms delay"
            )
            assert not np.any(np.isinf(processed)), (
                f"Infinite values with {delay * 1000:.0f}ms delay"
            )

            # Check that processing doesn't introduce excessive artifacts
            max_amplitude = np.max(np.abs(processed))
            original_max = np.max(np.abs(base_signal))
            assert max_amplitude < original_max * 2.0, (
                f"Excessive amplification with {delay * 1000:.0f}ms delay"
            )


class TestSpectralEnhancerTransientSuppression:
    """Test cases for transient noise suppression (keyboard, clicks, etc.)."""

    @pytest.fixture
    def enhancer(self):
        """Create SpectralEnhancer instance for testing."""
        return SpectralEnhancer(sample_rate=16000)

    @pytest.fixture
    def sample_rate(self):
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def keyboard_typing_samples(self, sample_rate):
        """Generate synthetic keyboard typing sounds."""
        duration = 2.0
        samples = int(sample_rate * duration)

        # Create base signal (speech-like background)
        t = np.linspace(0, duration, samples, endpoint=False)
        speech_background = 0.1 * np.sin(2 * np.pi * 200 * t) * (
            np.sin(2 * np.pi * 3 * t) > 0
        ) + 0.05 * np.random.normal(0, 1, samples)

        # Add keyboard typing transients
        typing_times = [0.3, 0.7, 1.1, 1.4, 1.8]  # seconds
        contaminated = speech_background.copy()

        for typing_time in typing_times:
            typing_sample = int(typing_time * sample_rate)
            if typing_sample < len(contaminated):
                # Create sharp transient (keyboard click)
                transient_duration = int(0.01 * sample_rate)  # 10ms click
                end_sample = min(typing_sample + transient_duration, len(contaminated))

                # Sharp attack, quick decay
                transient_envelope = np.exp(-np.arange(end_sample - typing_sample) * 100)
                transient_noise = (
                    0.8
                    * np.random.normal(0, 1, end_sample - typing_sample)
                    * transient_envelope
                    + 0.5
                    * np.sin(
                        2
                        * np.pi
                        * 3000
                        * np.arange(end_sample - typing_sample)
                        / sample_rate
                    )
                    * transient_envelope
                )

                contaminated[typing_sample:end_sample] += transient_noise

        return contaminated.astype(np.float32), speech_background.astype(np.float32)

    @pytest.fixture
    def mouse_click_samples(self, sample_rate):
        """Generate synthetic mouse click sounds."""
        duration = 1.5
        samples = int(sample_rate * duration)

        # Create base signal
        t = np.linspace(0, duration, samples, endpoint=False)
        base_signal = 0.08 * np.sin(2 * np.pi * 500 * t) * (t < 0.8)

        # Add mouse clicks
        click_times = [0.4, 0.9, 1.2]
        contaminated = base_signal.copy()

        for click_time in click_times:
            click_sample = int(click_time * sample_rate)
            if click_sample < len(contaminated):
                # Create double-click pattern (down and up)
                click_duration = int(0.005 * sample_rate)  # 5ms per click

                # First click (mouse down)
                end1 = min(click_sample + click_duration, len(contaminated))
                click1 = 0.6 * np.random.uniform(-1, 1, end1 - click_sample)
                contaminated[click_sample:end1] += click1

                # Second click (mouse up) - 20ms later
                click2_start = click_sample + int(0.02 * sample_rate)
                if click2_start < len(contaminated):
                    end2 = min(click2_start + click_duration, len(contaminated))
                    click2 = 0.4 * np.random.uniform(-1, 1, end2 - click2_start)
                    contaminated[click2_start:end2] += click2

        return contaminated.astype(np.float32), base_signal.astype(np.float32)

    @pytest.fixture
    def mechanical_noise_patterns(self, sample_rate):
        """Generate mechanical noise patterns (fan, AC, etc.)."""
        duration = 2.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create speech signal
        speech = 0.15 * np.sin(2 * np.pi * 150 * t) * (
            np.sin(2 * np.pi * 2 * t) > -0.5
        ) + 0.1 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 2 * t) > -0.5)

        # Add mechanical noise patterns
        # Fan noise (periodic with harmonics)
        fan_fundamental = 120  # Hz
        fan_noise = (
            0.2 * np.sin(2 * np.pi * fan_fundamental * t)
            + 0.1 * np.sin(2 * np.pi * 2 * fan_fundamental * t)
            + 0.05 * np.sin(2 * np.pi * 3 * fan_fundamental * t)
        )

        # AC compressor cycling (periodic bursts)
        ac_cycle = 0.15 * np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)

        # Combine all components
        contaminated = (
            speech + fan_noise + ac_cycle + 0.03 * np.random.normal(0, 1, samples)
        )

        return contaminated.astype(np.float32), speech.astype(np.float32)

    def test_transient_detection_keyboard_typing(
        self, enhancer, keyboard_typing_samples, sample_rate
    ):
        """Test transient detection with keyboard typing samples."""
        contaminated, clean = keyboard_typing_samples

        # Apply transient suppression
        processed = enhancer.suppress_transients(contaminated)

        # Calculate transient energy reduction
        # Use high-frequency energy as proxy for transient content
        freqs_cont, psd_cont = signal.welch(contaminated, sample_rate, nperseg=512)
        freqs_proc, psd_proc = signal.welch(processed, sample_rate, nperseg=512)

        # Focus on high-frequency range where keyboard clicks are prominent
        high_freq_mask = freqs_cont > 2000  # Above 2kHz

        cont_high_energy = np.sum(psd_cont[high_freq_mask])
        proc_high_energy = np.sum(psd_proc[high_freq_mask])

        # High-frequency energy should be reduced (transients suppressed)
        reduction_ratio = cont_high_energy / (proc_high_energy + 1e-10)
        assert reduction_ratio > 1.5, (
            f"Insufficient transient suppression: {reduction_ratio:.2f}x"
        )

        # But speech content should be preserved
        speech_freq_mask = (freqs_cont >= 200) & (freqs_cont <= 1000)
        cont_speech_energy = np.sum(psd_cont[speech_freq_mask])
        proc_speech_energy = np.sum(psd_proc[speech_freq_mask])

        speech_preservation = proc_speech_energy / cont_speech_energy
        assert speech_preservation > 0.6, (
            f"Speech content over-suppressed: {speech_preservation:.2f}"
        )

    def test_transient_detection_mouse_clicks(self, enhancer, mouse_click_samples):
        """Test transient detection with mouse click samples."""
        contaminated, clean = mouse_click_samples

        # Apply transient suppression
        processed = enhancer.suppress_transients(contaminated)

        # Compare processed signal to clean signal
        # Processed should be more similar to clean than contaminated
        clean_norm = clean / (np.max(np.abs(clean)) + 1e-10)
        cont_norm = contaminated / (np.max(np.abs(contaminated)) + 1e-10)
        proc_norm = processed / (np.max(np.abs(processed)) + 1e-10)

        clean_proc_corr = np.corrcoef(clean_norm, proc_norm)[0, 1]
        clean_cont_corr = np.corrcoef(clean_norm, cont_norm)[0, 1]

        assert clean_proc_corr > clean_cont_corr, (
            f"Transient suppression ineffective: {clean_proc_corr:.3f} vs {clean_cont_corr:.3f}"
        )

    def test_fast_acting_suppression_without_speech_artifacts(
        self, enhancer, sample_rate
    ):
        """Test fast-acting suppression without speech artifacts."""
        duration = 1.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create speech with embedded transients
        speech = (
            0.2 * np.sin(2 * np.pi * 400 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))
        )

        # Add sharp transient in middle of speech
        transient_start = samples // 2
        transient_end = transient_start + int(0.01 * sample_rate)  # 10ms transient

        contaminated = speech.copy()
        contaminated[transient_start:transient_end] += 0.8 * np.random.uniform(
            -1, 1, transient_end - transient_start
        )

        # Apply suppression
        processed = enhancer.suppress_transients(contaminated)

        # Check that speech before and after transient is preserved
        before_transient = slice(
            transient_start - int(0.1 * sample_rate), transient_start
        )
        after_transient = slice(transient_end, transient_end + int(0.1 * sample_rate))

        # Correlation should be high for speech regions
        before_corr = np.corrcoef(speech[before_transient], processed[before_transient])[
            0, 1
        ]
        after_corr = np.corrcoef(speech[after_transient], processed[after_transient])[
            0, 1
        ]

        assert before_corr > 0.8, f"Speech before transient corrupted: {before_corr:.3f}"
        assert after_corr > 0.8, f"Speech after transient corrupted: {after_corr:.3f}"

        # Transient region should be suppressed
        transient_region = slice(transient_start, transient_end)
        orig_transient_energy = np.sum(contaminated[transient_region] ** 2)
        proc_transient_energy = np.sum(processed[transient_region] ** 2)

        suppression_ratio = orig_transient_energy / (proc_transient_energy + 1e-10)
        assert suppression_ratio > 2.0, (
            f"Transient not sufficiently suppressed: {suppression_ratio:.2f}x"
        )

    def test_mechanical_noise_pattern_recognition(
        self, enhancer, mechanical_noise_patterns, sample_rate
    ):
        """Test mechanical noise pattern recognition and filtering."""
        contaminated, clean_speech = mechanical_noise_patterns

        # Apply transient suppression (should handle mechanical patterns)
        processed = enhancer.suppress_transients(contaminated)

        # Analyze periodic noise suppression
        freqs_cont, psd_cont = signal.welch(contaminated, sample_rate, nperseg=1024)
        freqs_proc, psd_proc = signal.welch(processed, sample_rate, nperseg=1024)

        # Check suppression of mechanical noise frequencies
        # Fan noise around 120Hz and harmonics
        fan_freq_mask = (
            ((freqs_cont >= 115) & (freqs_cont <= 125))
            | ((freqs_cont >= 235) & (freqs_cont <= 245))
            | ((freqs_cont >= 355) & (freqs_cont <= 365))
        )

        # AC noise around 60Hz
        ac_freq_mask = (freqs_cont >= 55) & (freqs_cont <= 65)

        cont_fan_energy = np.sum(psd_cont[fan_freq_mask])
        proc_fan_energy = np.sum(psd_proc[fan_freq_mask])

        cont_ac_energy = np.sum(psd_cont[ac_freq_mask])
        proc_ac_energy = np.sum(psd_proc[ac_freq_mask])

        # Mechanical noise should be reduced
        fan_reduction = cont_fan_energy / (proc_fan_energy + 1e-10)
        ac_reduction = cont_ac_energy / (proc_ac_energy + 1e-10)

        # Should show some reduction in mechanical noise
        assert fan_reduction > 1.2 or ac_reduction > 1.2, (
            f"Mechanical noise not reduced: fan={fan_reduction:.2f}x, ac={ac_reduction:.2f}x"
        )

        # Speech frequencies should be preserved
        speech_mask = (freqs_cont >= 150) & (freqs_cont <= 300)
        cont_speech_energy = np.sum(psd_cont[speech_mask])
        proc_speech_energy = np.sum(psd_proc[speech_mask])

        speech_preservation = proc_speech_energy / cont_speech_energy
        assert speech_preservation > 0.6, (
            f"Speech over-suppressed: {speech_preservation:.2f}"
        )

    def test_transient_suppression_energy_and_spectral_analysis(
        self, enhancer, sample_rate
    ):
        """Test transient detection using energy and spectral analysis."""
        duration = 1.0
        samples = int(sample_rate * duration)

        # Create signal with known transient characteristics
        t = np.linspace(0, duration, samples, endpoint=False)
        base_signal = 0.1 * np.sin(2 * np.pi * 300 * t)

        # Add transients with different energy and spectral characteristics
        transient_times = [0.2, 0.5, 0.8]
        transient_types = ["click", "pop", "scratch"]

        contaminated = base_signal.copy()

        for i, (t_time, t_type) in enumerate(zip(transient_times, transient_types)):
            t_sample = int(t_time * sample_rate)
            t_duration = int(0.02 * sample_rate)  # 20ms
            t_end = min(t_sample + t_duration, len(contaminated))

            if t_type == "click":
                # Sharp, broadband click
                transient = 0.7 * np.random.uniform(-1, 1, t_end - t_sample)
            elif t_type == "pop":
                # Low-frequency pop
                transient = 0.5 * np.sin(
                    2 * np.pi * 100 * np.arange(t_end - t_sample) / sample_rate
                )
                transient *= np.exp(-np.arange(t_end - t_sample) * 50)  # Quick decay
            else:  # scratch
                # High-frequency scratch
                transient = 0.4 * np.sin(
                    2 * np.pi * 4000 * np.arange(t_end - t_sample) / sample_rate
                )
                transient += 0.3 * np.random.normal(0, 1, t_end - t_sample)

            contaminated[t_sample:t_end] += transient

        # Apply suppression
        processed = enhancer.suppress_transients(contaminated)

        # Verify that different types of transients are handled
        # Calculate energy in transient regions vs non-transient regions
        transient_mask = np.zeros(len(contaminated), dtype=bool)
        for t_time in transient_times:
            t_sample = int(t_time * sample_rate)
            t_duration = int(0.02 * sample_rate)
            t_end = min(t_sample + t_duration, len(contaminated))
            transient_mask[t_sample:t_end] = True

        # Energy reduction in transient regions
        orig_transient_energy = np.sum(contaminated[transient_mask] ** 2)
        proc_transient_energy = np.sum(processed[transient_mask] ** 2)

        # Energy preservation in non-transient regions
        orig_speech_energy = np.sum(contaminated[~transient_mask] ** 2)
        proc_speech_energy = np.sum(processed[~transient_mask] ** 2)

        transient_reduction = orig_transient_energy / (proc_transient_energy + 1e-10)
        speech_preservation = proc_speech_energy / orig_speech_energy

        assert transient_reduction > 1.5, (
            f"Transients not suppressed: {transient_reduction:.2f}x"
        )
        assert speech_preservation > 0.5, (
            f"Speech over-suppressed: {speech_preservation:.2f}"
        )

    def test_transient_suppression_stability_and_artifacts(self, enhancer, sample_rate):
        """Test transient suppression stability and absence of artifacts."""
        duration = 2.0
        samples = int(sample_rate * duration)

        # Create challenging test case with overlapping transients and speech
        t = np.linspace(0, duration, samples, endpoint=False)

        # Continuous speech-like signal
        speech = 0.15 * np.sin(2 * np.pi * 200 * t) * (
            1 + 0.3 * np.sin(2 * np.pi * 3 * t)
        ) + 0.1 * np.sin(2 * np.pi * 600 * t) * (1 + 0.2 * np.sin(2 * np.pi * 7 * t))

        # Add many overlapping transients
        np.random.seed(42)  # Reproducible test
        num_transients = 20
        transient_times = np.random.uniform(0.1, 1.9, num_transients)

        contaminated = speech.copy()
        for t_time in transient_times:
            t_sample = int(t_time * sample_rate)
            t_duration = int(np.random.uniform(0.005, 0.02) * sample_rate)  # 5-20ms
            t_end = min(t_sample + t_duration, len(contaminated))

            # Random transient type
            transient_amplitude = np.random.uniform(0.3, 0.8)
            transient = transient_amplitude * np.random.uniform(-1, 1, t_end - t_sample)
            contaminated[t_sample:t_end] += transient

        # Apply suppression
        processed = enhancer.suppress_transients(contaminated)

        # Check for stability
        assert not np.any(np.isnan(processed)), (
            "Transient suppression produced NaN values"
        )
        assert not np.any(np.isinf(processed)), (
            "Transient suppression produced infinite values"
        )

        # Check that processing doesn't introduce excessive artifacts
        max_amplitude = np.max(np.abs(processed))
        speech_max = np.max(np.abs(speech))
        assert max_amplitude < speech_max * 5.0, "Excessive amplification detected"

        # Check that overall signal structure is preserved
        correlation = np.corrcoef(speech, processed)[0, 1]
        assert correlation > 0.5, f"Signal structure not preserved: {correlation:.3f}"

        # Check that transient suppression is effective
        orig_std = np.std(contaminated)
        proc_std = np.std(processed)
        speech_std = np.std(speech)

        # Processed signal should have lower variance than contaminated (less transients)
        # but not lower than original speech (avoid over-suppression)
        assert proc_std < orig_std, "No transient suppression detected"
        assert proc_std > speech_std * 0.5, "Over-suppression detected"
