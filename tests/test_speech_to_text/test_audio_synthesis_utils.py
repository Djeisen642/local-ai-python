"""Utilities for synthesizing test audio with various noise conditions."""

import io
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


class AudioSynthesizer:
    """Utility class for creating synthetic audio with controlled noise conditions."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio synthesizer.

        Args:
            sample_rate: Sample rate for generated audio
        """
        self.sample_rate = sample_rate

    def load_clean_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load clean audio from a WAV file.

        Args:
            file_path: Path to the WAV file

        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        with wave.open(str(file_path), "rb") as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()

            # Convert to numpy array
            if wav_file.getsampwidth() == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16)
            else:  # Assume 32-bit float
                audio_data = np.frombuffer(frames, dtype=np.float32)

            # Convert to mono if stereo
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                audio_data = np.mean(audio_data, axis=1)

            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0

            return audio_data, sample_rate

    def add_white_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add white noise to audio at specified SNR.

        Args:
            audio: Clean audio samples
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Audio with added white noise
        """
        # Calculate signal power
        signal_power = np.mean(audio**2)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate white noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

        return audio + noise

    def add_pink_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add pink noise (1/f noise) to audio at specified SNR.

        Args:
            audio: Clean audio samples
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Audio with added pink noise
        """
        # Generate white noise
        white_noise = np.random.normal(0, 1, len(audio))

        # Apply pink noise filter (approximate)
        # Pink noise has -3dB/octave rolloff
        fft_noise = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(len(white_noise), 1 / self.sample_rate)

        # Create pink noise filter (1/f characteristic)
        pink_filter = np.ones_like(freqs)
        pink_filter[1:] = 1 / np.sqrt(np.abs(freqs[1:]))
        pink_filter[0] = pink_filter[1]  # Avoid division by zero

        # Apply filter and convert back to time domain
        pink_noise_fft = fft_noise * pink_filter
        pink_noise = np.real(np.fft.ifft(pink_noise_fft))

        # Scale to desired SNR
        signal_power = np.mean(audio**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Normalize and scale noise
        pink_noise = pink_noise / np.sqrt(np.mean(pink_noise**2))
        pink_noise = pink_noise * np.sqrt(noise_power)

        return audio + pink_noise

    def add_hum_noise(
        self, audio: np.ndarray, frequency: float = 60.0, snr_db: float = 20.0
    ) -> np.ndarray:
        """
        Add electrical hum noise (sine wave) to audio.

        Args:
            audio: Clean audio samples
            frequency: Hum frequency in Hz (typically 50Hz or 60Hz)
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Audio with added hum noise
        """
        # Generate time vector
        t = np.arange(len(audio)) / self.sample_rate

        # Generate hum signal
        hum = np.sin(2 * np.pi * frequency * t)

        # Scale to desired SNR
        signal_power = np.mean(audio**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        hum = hum * np.sqrt(noise_power)

        return audio + hum

    def add_click_noise(
        self, audio: np.ndarray, click_rate: float = 2.0, click_amplitude: float = 0.5
    ) -> np.ndarray:
        """
        Add random click/pop noises to simulate keyboard typing or mouse clicks.

        Args:
            audio: Clean audio samples
            click_rate: Average clicks per second
            click_amplitude: Amplitude of clicks relative to signal

        Returns:
            Audio with added click noise
        """
        audio_with_clicks = audio.copy()

        # Calculate number of clicks based on duration and rate
        duration = len(audio) / self.sample_rate
        num_clicks = int(duration * click_rate)

        # Generate random click positions
        click_positions = np.random.randint(0, len(audio), num_clicks)

        # Add clicks at random positions
        for pos in click_positions:
            # Create a short impulse (click)
            click_duration = int(0.001 * self.sample_rate)  # 1ms click
            click_end = min(pos + click_duration, len(audio))

            # Generate click waveform (decaying impulse)
            click_samples = np.arange(click_end - pos)
            click_waveform = click_amplitude * np.exp(
                -click_samples / (click_duration * 0.3)
            )

            # Add random polarity
            if np.random.random() > 0.5:
                click_waveform = -click_waveform

            audio_with_clicks[pos:click_end] += click_waveform

        return audio_with_clicks

    def add_room_reverb(
        self, audio: np.ndarray, reverb_time: float = 0.3, wet_level: float = 0.3
    ) -> np.ndarray:
        """
        Add simple room reverberation to audio.

        Args:
            audio: Clean audio samples
            reverb_time: Reverberation time in seconds
            wet_level: Amount of reverb to mix (0.0 to 1.0)

        Returns:
            Audio with added reverberation
        """
        # Create simple reverb using multiple delayed copies
        reverb_delays = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]  # Delays in seconds
        reverb_gains = [0.6, 0.4, 0.3, 0.2, 0.15, 0.1]  # Corresponding gains

        reverb_audio = audio.copy()

        for delay, gain in zip(reverb_delays, reverb_gains):
            if delay > reverb_time:
                break

            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(audio):
                # Create delayed version
                delayed_audio = np.zeros_like(audio)
                delayed_audio[delay_samples:] = audio[:-delay_samples]

                # Add to reverb with decay
                decay_factor = np.exp(-delay / reverb_time)
                reverb_audio += delayed_audio * gain * decay_factor * wet_level

        return reverb_audio

    def save_audio_as_wav(
        self, audio: np.ndarray, file_path: Path, sample_rate: int = None
    ) -> None:
        """
        Save audio samples as a WAV file.

        Args:
            audio: Audio samples to save
            file_path: Output file path
            sample_rate: Sample rate (uses instance default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Ensure output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize and convert to 16-bit integers
        audio_normalized = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def create_noisy_test_suite(self, clean_audio_path: Path, output_dir: Path) -> None:
        """
        Create a comprehensive test suite with various noise conditions.

        Args:
            clean_audio_path: Path to clean reference audio
            output_dir: Directory to save noisy audio files
        """
        # Load clean audio
        clean_audio, original_sample_rate = self.load_clean_audio(clean_audio_path)

        # Resample if needed
        if original_sample_rate != self.sample_rate:
            # Simple resampling (for test purposes)
            ratio = self.sample_rate / original_sample_rate
            new_length = int(len(clean_audio) * ratio)
            clean_audio = np.interp(
                np.linspace(0, len(clean_audio), new_length),
                np.arange(len(clean_audio)),
                clean_audio,
            )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save clean reference
        self.save_audio_as_wav(clean_audio, output_dir / "clean.wav")

        # Generate various noise conditions
        noise_conditions = [
            ("white_noise_20db", lambda x: self.add_white_noise(x, 20.0)),
            ("white_noise_10db", lambda x: self.add_white_noise(x, 10.0)),
            ("white_noise_5db", lambda x: self.add_white_noise(x, 5.0)),
            ("pink_noise_15db", lambda x: self.add_pink_noise(x, 15.0)),
            ("pink_noise_8db", lambda x: self.add_pink_noise(x, 8.0)),
            ("hum_60hz", lambda x: self.add_hum_noise(x, 60.0, 25.0)),
            ("hum_50hz", lambda x: self.add_hum_noise(x, 50.0, 25.0)),
            ("clicks_light", lambda x: self.add_click_noise(x, 1.0, 0.3)),
            ("clicks_heavy", lambda x: self.add_click_noise(x, 3.0, 0.5)),
            ("reverb_light", lambda x: self.add_room_reverb(x, 0.2, 0.2)),
            ("reverb_heavy", lambda x: self.add_room_reverb(x, 0.5, 0.4)),
        ]

        # Generate combined noise conditions
        combined_conditions = [
            (
                "white_pink_combo",
                lambda x: self.add_pink_noise(self.add_white_noise(x, 15.0), 12.0),
            ),
            (
                "hum_clicks_combo",
                lambda x: self.add_click_noise(
                    self.add_hum_noise(x, 60.0, 25.0), 2.0, 0.3
                ),
            ),
            (
                "reverb_noise_combo",
                lambda x: self.add_white_noise(self.add_room_reverb(x, 0.3, 0.3), 12.0),
            ),
        ]

        # Generate all noise conditions
        all_conditions = noise_conditions + combined_conditions

        for condition_name, noise_func in all_conditions:
            try:
                noisy_audio = noise_func(clean_audio)
                output_path = output_dir / f"{condition_name}.wav"
                self.save_audio_as_wav(noisy_audio, output_path)
                print(f"Generated: {output_path}")
            except Exception as e:
                print(f"Error generating {condition_name}: {e}")


def create_synthetic_test_audio():
    """Create synthetic test audio files for accuracy validation."""
    synthesizer = AudioSynthesizer(sample_rate=16000)

    # Define test data directory
    test_data_dir = Path(__file__).parent.parent / "test_data" / "audio"
    synthetic_dir = test_data_dir / "synthetic_noise"

    # Use existing clean audio files as base
    clean_files = [
        "hello_world.wav",
        "short_sentence.wav",
        "numbers.wav",
    ]

    for clean_file in clean_files:
        clean_path = test_data_dir / clean_file
        if clean_path.exists():
            # Extract filename without extension
            file_stem = Path(clean_file).stem
            output_dir = synthetic_dir / file_stem
            synthesizer.create_noisy_test_suite(clean_path, output_dir)
            print(f"Created synthetic test suite for {clean_file}")
        else:
            print(f"Clean audio file not found: {clean_path}")


if __name__ == "__main__":
    create_synthetic_test_audio()
