"""Integration tests for VoiceActivityDetector using real audio files."""

import wave
from pathlib import Path

import pytest

from local_ai.speech_to_text.vad import VoiceActivityDetector


class TestVoiceActivityDetectorIntegration:
    """Integration test cases for VoiceActivityDetector using real audio files."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test data directory path."""
        return Path(__file__).parent.parent / "test_data" / "audio"

    @pytest.fixture
    def vad(self) -> VoiceActivityDetector:
        """Create a VoiceActivityDetector instance for testing."""
        return VoiceActivityDetector()

    def load_wav_file(self, file_path: Path) -> bytes:
        """
        Load a WAV file and return raw audio data.

        Args:
            file_path: Path to the WAV file

        Returns:
            Raw audio data as bytes
        """
        with wave.open(str(file_path), "rb") as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            # Read all frames
            audio_data = wav_file.readframes(frames)

            # Convert to mono if stereo (take left channel)
            if channels == 2:
                # Convert stereo to mono by taking every other sample pair
                mono_data = bytearray()
                for i in range(
                    0, len(audio_data), 4
                ):  # 4 bytes = 2 samples * 2 bytes each
                    # Take left channel (first 2 bytes of each 4-byte group)
                    mono_data.extend(audio_data[i : i + 2])
                audio_data = bytes(mono_data)

            return audio_data, sample_rate, sample_width

    def chunk_audio(self, audio_data: bytes, vad: VoiceActivityDetector) -> list[bytes]:
        """
        Split audio data into chunks suitable for VAD processing.

        Args:
            audio_data: Raw audio data
            vad: VoiceActivityDetector instance

        Returns:
            List of audio chunks
        """
        chunk_size = vad.frame_size * 2  # 2 bytes per sample (16-bit)
        chunks = []

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            if len(chunk) == chunk_size:  # Only use complete chunks
                chunks.append(chunk)

        return chunks

    @pytest.mark.integration
    def test_silence_detection(
        self, vad: VoiceActivityDetector, test_data_dir: Path
    ) -> None:
        """Test that silence is correctly identified as non-speech."""
        silence_file = test_data_dir / "edge_cases" / "silence.wav"

        if not silence_file.exists():
            pytest.skip(f"Silence test file not found: {silence_file}")

        # Load silence audio
        audio_data, sample_rate, _ = self.load_wav_file(silence_file)

        # Create VAD with matching sample rate if needed
        if sample_rate in [8000, 16000, 32000, 48000]:
            test_vad = VoiceActivityDetector(sample_rate=sample_rate)
        else:
            test_vad = vad  # Use default 16kHz

        # Split into chunks
        chunks = self.chunk_audio(audio_data, test_vad)

        # Test individual chunks - most should be silence
        silence_count = 0
        for chunk in chunks:
            if not test_vad.is_speech(chunk):
                silence_count += 1

        # At least 80% of chunks should be detected as silence
        silence_ratio = silence_count / len(chunks) if chunks else 1.0
        assert silence_ratio >= 0.8, (
            f"Expected mostly silence, got {silence_ratio:.2%} silence"
        )

        # Test get_speech_segments - should return very few or no segments
        speech_segments = test_vad.get_speech_segments(chunks)
        speech_ratio = len(speech_segments) / len(chunks) if chunks else 0.0
        assert speech_ratio <= 0.2, (
            f"Expected little speech in silence, got {speech_ratio:.2%} speech"
        )

    @pytest.mark.integration
    def test_speech_detection(
        self, vad: VoiceActivityDetector, test_data_dir: Path
    ) -> None:
        """Test that clear speech is correctly identified."""
        speech_files = ["hello_world.wav", "short_sentence.wav", "numbers.wav"]

        for filename in speech_files:
            speech_file = test_data_dir / filename

            if not speech_file.exists():
                pytest.skip(f"Speech test file not found: {speech_file}")
                continue

            # Load speech audio
            audio_data, sample_rate, _ = self.load_wav_file(speech_file)

            # Create VAD with matching sample rate if needed
            if sample_rate in [8000, 16000, 32000, 48000]:
                test_vad = VoiceActivityDetector(sample_rate=sample_rate)
            else:
                test_vad = vad  # Use default 16kHz

            # Split into chunks
            chunks = self.chunk_audio(audio_data, test_vad)

            if not chunks:
                pytest.skip(f"No valid chunks found in {filename}")
                continue

            # Test get_speech_segments - should find some speech
            speech_segments = test_vad.get_speech_segments(chunks)
            speech_ratio = len(speech_segments) / len(chunks)

            # Expect at least some speech detection (at least 10% of chunks)
            assert speech_ratio >= 0.1, (
                f"Expected some speech in {filename}, got {speech_ratio:.2%} speech"
            )

    @pytest.mark.integration
    def test_mixed_audio_segmentation(
        self, vad: VoiceActivityDetector, test_data_dir: Path
    ) -> None:
        """Test speech segmentation on audio with both speech and silence."""
        # Use a longer file that likely has some silence at the beginning/end
        test_file = test_data_dir / "short_sentence.wav"

        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        # Load audio
        audio_data, sample_rate, _ = self.load_wav_file(test_file)

        # Create VAD with matching sample rate if needed
        if sample_rate in [8000, 16000, 32000, 48000]:
            test_vad = VoiceActivityDetector(sample_rate=sample_rate)
        else:
            test_vad = vad

        # Split into chunks
        chunks = self.chunk_audio(audio_data, test_vad)

        if not chunks:
            pytest.skip("No valid chunks found")

        # Get speech segments
        speech_segments = test_vad.get_speech_segments(chunks)

        # Basic sanity checks
        assert isinstance(speech_segments, list), "Should return a list"
        assert len(speech_segments) <= len(chunks), (
            "Cannot have more speech segments than total chunks"
        )

        # All speech segments should be valid audio chunks
        for segment in speech_segments:
            assert isinstance(segment, bytes), "Each segment should be bytes"
            assert len(segment) == test_vad.frame_size * 2, (
                "Each segment should be the correct size"
            )

    @pytest.mark.integration
    def test_different_sample_rates(self, test_data_dir: Path) -> None:
        """Test VAD with different sample rates using quality test files."""
        quality_dir = test_data_dir / "quality"

        test_cases = [
            ("high_quality_16khz.wav", 16000),
            ("low_quality_8khz.wav", 8000),
        ]

        for filename, expected_rate in test_cases:
            test_file = quality_dir / filename

            if not test_file.exists():
                pytest.skip(f"Quality test file not found: {test_file}")
                continue

            # Create VAD with matching sample rate
            test_vad = VoiceActivityDetector(sample_rate=expected_rate)

            # Load audio
            audio_data, actual_rate, _ = self.load_wav_file(test_file)

            # Skip if sample rate doesn't match expected
            if actual_rate != expected_rate:
                pytest.skip(
                    f"Sample rate mismatch: expected {expected_rate}, got {actual_rate}"
                )
                continue

            # Split into chunks
            chunks = self.chunk_audio(audio_data, test_vad)

            if not chunks:
                pytest.skip(f"No valid chunks found in {filename}")
                continue

            # Test that VAD works with this sample rate
            speech_segments = test_vad.get_speech_segments(chunks)

            # Should detect some speech in "Hello world" audio
            speech_ratio = len(speech_segments) / len(chunks)
            assert speech_ratio >= 0.1, (
                f"Expected some speech detection at {expected_rate}Hz"
            )

    @pytest.mark.integration
    def test_very_short_audio(
        self, vad: VoiceActivityDetector, test_data_dir: Path
    ) -> None:
        """Test VAD behavior with very short audio files."""
        short_file = test_data_dir / "edge_cases" / "very_short.wav"

        if not short_file.exists():
            pytest.skip(f"Short audio test file not found: {short_file}")

        # Load short audio
        audio_data, sample_rate, _ = self.load_wav_file(short_file)

        # Create VAD with matching sample rate if needed
        if sample_rate in [8000, 16000, 32000, 48000]:
            test_vad = VoiceActivityDetector(sample_rate=sample_rate)
        else:
            test_vad = vad

        # Split into chunks
        chunks = self.chunk_audio(audio_data, test_vad)

        # Even very short audio should be processable
        speech_segments = test_vad.get_speech_segments(chunks)

        # Basic sanity checks
        assert isinstance(speech_segments, list), (
            "Should return a list even for short audio"
        )
        assert len(speech_segments) <= len(chunks), (
            "Cannot have more segments than chunks"
        )

    @pytest.mark.integration
    def test_scenario_audio_files(
        self, vad: VoiceActivityDetector, test_data_dir: Path
    ) -> None:
        """Test VAD on different speech scenarios."""
        scenarios_dir = test_data_dir / "scenarios"

        scenario_files = [
            "command.wav",  # "Start recording now"
            "dictation.wav",  # "Please transcribe this message"
            "question.wav",  # "What time is it"
        ]

        for filename in scenario_files:
            scenario_file = scenarios_dir / filename

            if not scenario_file.exists():
                pytest.skip(f"Scenario test file not found: {scenario_file}")
                continue

            # Load scenario audio
            audio_data, sample_rate, _ = self.load_wav_file(scenario_file)

            # Create VAD with matching sample rate if needed
            if sample_rate in [8000, 16000, 32000, 48000]:
                test_vad = VoiceActivityDetector(sample_rate=sample_rate)
            else:
                test_vad = vad

            # Split into chunks
            chunks = self.chunk_audio(audio_data, test_vad)

            if not chunks:
                pytest.skip(f"No valid chunks found in {filename}")
                continue

            # Test speech detection
            speech_segments = test_vad.get_speech_segments(chunks)
            speech_ratio = len(speech_segments) / len(chunks)

            # Each scenario should have some detectable speech
            assert speech_ratio >= 0.1, (
                f"Expected speech detection in {filename}, got {speech_ratio:.2%}"
            )

            # Log results for debugging
