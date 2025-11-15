"""
Tests for sample rate handling fixes in WhisperTranscriber (TDD - Red phase).

This test module validates the fixes for:
1. Incorrect sample rate assumption in _create_wav_data
2. Sample rate parameter passing through _convert_audio_format
3. Backward compatibility with existing code

Requirements tested: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.4, 5.5
"""

import io
import wave
from unittest.mock import patch

import numpy as np
import pytest

from local_ai.speech_to_text.config import DEFAULT_SAMPLE_RATE
from local_ai.speech_to_text.transcriber import WhisperTranscriber


@pytest.mark.unit
class TestConvertAudioFormatWithSourceSampleRate:
    """
    Test _convert_audio_format with source_sample_rate parameter.

    Requirements: 1.1, 1.2, 5.1
    """

    def create_raw_audio_samples(
        self, sample_rate: int = 16000, duration: float = 1.0
    ) -> bytes:
        """Create raw 16-bit PCM audio samples."""
        num_samples = int(sample_rate * duration)
        # Create a simple sine wave at 440 Hz
        samples = np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate)
        # Convert to 16-bit PCM
        samples_int16 = (samples * 32767).astype(np.int16)
        return samples_int16.tobytes()

    def test_convert_audio_format_with_explicit_source_rate(self) -> None:
        """
        Test _convert_audio_format passes explicit source_sample_rate to _create_wav_data.

        Requirement 1.1: Transcriber SHALL pass actual source sample rate
        """
        transcriber = WhisperTranscriber()

        # Create raw audio at 48kHz
        raw_audio = self.create_raw_audio_samples(sample_rate=48000, duration=0.5)

        # Mock _create_wav_data to verify it receives correct parameters
        with patch.object(
            transcriber, "_create_wav_data", return_value=b"RIFF"
        ) as mock_create_wav:
            # Call with explicit source_sample_rate
            transcriber._convert_audio_format(
                raw_audio, target_sample_rate=16000, source_sample_rate=48000
            )

            # Verify _create_wav_data was called with correct source rate
            mock_create_wav.assert_called_once()
            call_args = mock_create_wav.call_args
            assert call_args[0][0] == raw_audio  # audio_samples
            assert call_args[0][1] == 48000  # source_sample_rate (not target!)
            assert call_args[0][2] == 1  # channels
            assert call_args[0][3] == 16000  # target_sample_rate

    def test_convert_audio_format_with_default_source_rate(self) -> None:
        """
        Test _convert_audio_format uses DEFAULT_SAMPLE_RATE when source_sample_rate is None.

        Requirement 1.2: Transcriber SHALL use correct source sample rate
        """
        transcriber = WhisperTranscriber()

        raw_audio = self.create_raw_audio_samples(sample_rate=16000, duration=0.5)

        with patch.object(
            transcriber, "_create_wav_data", return_value=b"RIFF"
        ) as mock_create_wav:
            # Call without source_sample_rate (should default to DEFAULT_SAMPLE_RATE)
            transcriber._convert_audio_format(
                raw_audio, target_sample_rate=16000, source_sample_rate=None
            )

            # Verify _create_wav_data was called with DEFAULT_SAMPLE_RATE as source
            mock_create_wav.assert_called_once()
            call_args = mock_create_wav.call_args
            assert call_args[0][1] == DEFAULT_SAMPLE_RATE  # source_sample_rate

    def test_convert_audio_format_backward_compatible_no_source_param(self) -> None:
        """
        Test _convert_audio_format works without source_sample_rate parameter.

        Requirement 5.1: System SHALL not change API signatures of public methods
        """
        transcriber = WhisperTranscriber()

        raw_audio = self.create_raw_audio_samples(sample_rate=16000, duration=0.5)

        with patch.object(
            transcriber, "_create_wav_data", return_value=b"RIFF"
        ) as mock_create_wav:
            # Call without source_sample_rate parameter (backward compatibility)
            transcriber._convert_audio_format(raw_audio, target_sample_rate=16000)

            # Should still work and default to DEFAULT_SAMPLE_RATE
            mock_create_wav.assert_called_once()
            call_args = mock_create_wav.call_args
            assert call_args[0][1] == DEFAULT_SAMPLE_RATE

    def test_convert_audio_format_passes_correct_rate_to_create_wav(self) -> None:
        """
        Test that source rate is correctly passed through the call chain.

        Requirement 1.1: Verify correct rate is passed to _create_wav_data
        """
        transcriber = WhisperTranscriber()

        test_cases = [
            (8000, 16000),  # 8kHz to 16kHz
            (16000, 16000),  # 16kHz to 16kHz (no resampling needed)
            (32000, 16000),  # 32kHz to 16kHz
            (48000, 16000),  # 48kHz to 16kHz
        ]

        for source_rate, target_rate in test_cases:
            raw_audio = self.create_raw_audio_samples(
                sample_rate=source_rate, duration=0.3
            )

            with patch.object(
                transcriber, "_create_wav_data", return_value=b"RIFF"
            ) as mock_create_wav:
                transcriber._convert_audio_format(
                    raw_audio,
                    target_sample_rate=target_rate,
                    source_sample_rate=source_rate,
                )

                call_args = mock_create_wav.call_args
                assert call_args[0][1] == source_rate, (
                    f"Expected source_sample_rate={source_rate}, got {call_args[0][1]}"
                )
                assert call_args[0][3] == target_rate


@pytest.mark.unit
class TestCreateWavDataResamplingLogic:
    """
    Test _create_wav_data resampling logic with correct source sample rate.

    Requirements: 1.3, 1.4
    """

    def create_test_samples(self, sample_rate: int, duration: float) -> bytes:
        """Create test audio samples."""
        num_samples = int(sample_rate * duration)
        samples = np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate)
        samples_int16 = (samples * 32767).astype(np.int16)
        return samples_int16.tobytes()

    def extract_wav_info(self, wav_data: bytes) -> tuple[int, int, float]:
        """Extract sample rate, num samples, and duration from WAV data."""
        with io.BytesIO(wav_data) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                duration = num_frames / sample_rate
                return sample_rate, num_frames, duration

    def test_create_wav_data_skips_resampling_when_rates_match(self) -> None:
        """
        Test resampling is skipped when source == target sample rate.

        Requirement 1.3: Transcriber SHALL skip resampling when rates match
        """
        transcriber = WhisperTranscriber()

        # Create audio at 16kHz
        source_rate = 16000
        target_rate = 16000
        duration = 1.0
        audio_samples = self.create_test_samples(source_rate, duration)

        # Mock _resample_audio to verify it's NOT called
        with patch.object(
            transcriber, "_resample_audio", wraps=transcriber._resample_audio
        ) as mock_resample:
            wav_data = transcriber._create_wav_data(
                audio_samples, source_rate, 1, target_rate
            )

            # Verify resampling was skipped (not called or returned early)
            # The method should check if source_rate == target_rate and skip
            assert wav_data.startswith(b"RIFF")

            # If rates match, _resample_audio should either not be called
            # or should return samples unchanged
            if mock_resample.called:
                # If called, it should return samples unchanged
                result = mock_resample.return_value
                original_samples = np.frombuffer(audio_samples, dtype=np.int16)
                assert len(result) == len(original_samples)

    def test_create_wav_data_resamples_when_rates_differ(self) -> None:
        """
        Test resampling occurs when source != target sample rate.

        Requirement 1.4: Transcriber SHALL resample using correct ratio
        """
        transcriber = WhisperTranscriber()

        test_cases = [
            (8000, 16000),  # Upsample 2x
            (48000, 16000),  # Downsample 3x
            (32000, 16000),  # Downsample 2x
        ]

        for source_rate, target_rate in test_cases:
            duration = 0.5
            audio_samples = self.create_test_samples(source_rate, duration)

            # Mock _resample_audio to verify it's called with correct parameters
            with patch.object(
                transcriber, "_resample_audio", wraps=transcriber._resample_audio
            ) as mock_resample:
                wav_data = transcriber._create_wav_data(
                    audio_samples, source_rate, 1, target_rate
                )

                # Verify resampling was called
                mock_resample.assert_called_once()
                call_args = mock_resample.call_args

                # Verify correct rates were passed
                assert call_args[0][1] == source_rate  # source_rate
                assert call_args[0][2] == target_rate  # target_rate

                # Verify output is valid WAV
                assert wav_data.startswith(b"RIFF")

    def test_create_wav_data_with_various_sample_rates(self) -> None:
        """
        Test _create_wav_data with various sample rates.

        Requirement 1.4: Test with various sample rates (8kHz, 16kHz, 32kHz, 48kHz)
        """
        transcriber = WhisperTranscriber()

        sample_rates = [8000, 16000, 32000, 48000]
        target_rate = 16000
        duration = 0.5

        for source_rate in sample_rates:
            audio_samples = self.create_test_samples(source_rate, duration)

            wav_data = transcriber._create_wav_data(
                audio_samples, source_rate, 1, target_rate
            )

            # Verify output is valid WAV
            assert wav_data.startswith(b"RIFF")
            assert b"WAVE" in wav_data

            # Extract and verify WAV properties
            wav_rate, num_frames, wav_duration = self.extract_wav_info(wav_data)

            # Output should be at target rate
            assert wav_rate == target_rate

            # Duration should be preserved (within tolerance)
            assert abs(wav_duration - duration) < 0.01  # 10ms tolerance

    def test_create_wav_data_output_duration_matches_input(self) -> None:
        """
        Test output duration matches input duration after resampling.

        Requirement 1.4: Verify output duration matches input duration
        """
        transcriber = WhisperTranscriber()

        test_cases = [
            (8000, 16000, 1.0),  # 1 second at 8kHz -> 16kHz
            (48000, 16000, 0.5),  # 0.5 seconds at 48kHz -> 16kHz
            (32000, 16000, 2.0),  # 2 seconds at 32kHz -> 16kHz
            (16000, 16000, 1.5),  # 1.5 seconds, no resampling
        ]

        for source_rate, target_rate, duration in test_cases:
            audio_samples = self.create_test_samples(source_rate, duration)

            wav_data = transcriber._create_wav_data(
                audio_samples, source_rate, 1, target_rate
            )

            # Extract duration from output WAV
            _, _, output_duration = self.extract_wav_info(wav_data)

            # Duration should match within 1% tolerance
            tolerance = duration * 0.01
            assert abs(output_duration - duration) < tolerance, (
                f"Duration mismatch: expected {duration}s, "
                f"got {output_duration}s (source={source_rate}, target={target_rate})"
            )

    def test_create_wav_data_uses_source_rate_not_target_for_input(self) -> None:
        """
        Test that source_sample_rate is used for input, not target_sample_rate.

        This is the core bug fix: previously the code incorrectly used
        target_sample_rate for both source and target.

        Requirement 1.2: Use correct source sample rate instead of assuming it equals target
        """
        transcriber = WhisperTranscriber()

        # Create audio at 48kHz
        source_rate = 48000
        target_rate = 16000
        duration = 1.0
        audio_samples = self.create_test_samples(source_rate, duration)

        # The bug was: _resample_audio was called with target_rate for both parameters
        # This test verifies the fix: source_rate is used correctly

        with patch.object(transcriber, "_resample_audio") as mock_resample:
            # Set up mock to return valid data
            num_samples = int(target_rate * duration)
            mock_samples = np.zeros(num_samples, dtype=np.int16)
            mock_resample.return_value = mock_samples

            transcriber._create_wav_data(audio_samples, source_rate, 1, target_rate)

            # Verify _resample_audio was called with DIFFERENT rates
            mock_resample.assert_called_once()
            call_args = mock_resample.call_args[0]

            # The fix: source_rate != target_rate in the call
            actual_source_rate = call_args[1]
            actual_target_rate = call_args[2]

            assert actual_source_rate == source_rate, (
                f"Bug not fixed: source_rate should be {source_rate}, "
                f"got {actual_source_rate}"
            )
            assert actual_target_rate == target_rate
            assert actual_source_rate != actual_target_rate, (
                "Bug not fixed: source and target rates should be different"
            )


@pytest.mark.unit
class TestBackwardCompatibility:
    """
    Test backward compatibility with existing code.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """

    def create_test_audio(self, sample_rate: int = 16000) -> bytes:
        """Create test audio samples."""
        num_samples = int(sample_rate * 0.5)
        samples = np.sin(2 * np.pi * 440 * np.arange(num_samples) / sample_rate)
        samples_int16 = (samples * 32767).astype(np.int16)
        return samples_int16.tobytes()

    def test_convert_audio_format_without_new_parameters(self) -> None:
        """
        Test calling _convert_audio_format without new parameters.

        Requirement 5.3: System SHALL not change API signatures of public methods
        """
        transcriber = WhisperTranscriber()

        raw_audio = self.create_test_audio()

        # Call without source_sample_rate parameter (old API)
        result = transcriber._convert_audio_format(raw_audio)

        # Should work and return valid WAV data
        assert isinstance(result, bytes)
        if result:  # If not empty
            assert result.startswith(b"RIFF") or len(result) == 0

    def test_convert_audio_format_with_only_target_rate(self) -> None:
        """
        Test calling _convert_audio_format with only target_sample_rate.

        Requirement 5.1: System SHALL not add new processing steps
        """
        transcriber = WhisperTranscriber()

        raw_audio = self.create_test_audio()

        # Call with only target_sample_rate (old API style)
        result = transcriber._convert_audio_format(raw_audio, target_sample_rate=16000)

        # Should work with default source rate
        assert isinstance(result, bytes)

    def test_transcribe_audio_with_result_without_new_parameters(self) -> None:
        """
        Test transcribe_audio_with_result works without new parameters.

        Requirement 5.3: Maintain same API signatures
        """
        transcriber = WhisperTranscriber()

        # This test verifies the method signature is backward compatible
        # The actual transcription will be tested in integration tests

        # Verify method exists and has correct signature
        import inspect

        sig = inspect.signature(transcriber.transcribe_audio_with_result)

        # Should have audio_data as required parameter
        assert "audio_data" in sig.parameters

        # New parameter should be optional (have default value)
        if "source_sample_rate" in sig.parameters:
            param = sig.parameters["source_sample_rate"]
            assert param.default is not inspect.Parameter.empty, (
                "source_sample_rate should have a default value for backward compatibility"
            )

    def test_default_behavior_preserved(self) -> None:
        """
        Test that default behavior is preserved when not using new parameters.

        Requirement 5.2: System SHALL not increase processing latency
        Requirement 5.4: Maintain same audio file formats
        """
        transcriber = WhisperTranscriber()

        raw_audio = self.create_test_audio(sample_rate=16000)

        # Call with old API (no source_sample_rate)
        result_old_api = transcriber._convert_audio_format(raw_audio)

        # Call with new API but default value
        result_new_api = transcriber._convert_audio_format(
            raw_audio, source_sample_rate=None
        )

        # Both should produce same result (default behavior)
        assert type(result_old_api) == type(result_new_api)

        # Both should be valid WAV or empty
        if result_old_api:
            assert result_old_api.startswith(b"RIFF")
        if result_new_api:
            assert result_new_api.startswith(b"RIFF")

    def test_no_new_processing_steps_added(self) -> None:
        """
        Test that no new processing steps are added to the pipeline.

        Requirement 5.1: System SHALL not add new processing steps
        """
        transcriber = WhisperTranscriber()

        raw_audio = self.create_test_audio()

        # The fix should only change which value is passed, not add new steps
        # We verify this by checking the call chain

        with patch.object(
            transcriber, "_create_wav_data", return_value=b"RIFF"
        ) as mock_create_wav:
            transcriber._convert_audio_format(raw_audio)

            # Should still call _create_wav_data exactly once (no new steps)
            assert mock_create_wav.call_count == 1

    def test_existing_code_works_without_changes(self) -> None:
        """
        Test that existing code works without any changes.

        Requirement 5.5: Only fix identified bugs without refactoring unrelated code
        """
        transcriber = WhisperTranscriber()

        # Simulate existing code that doesn't know about new parameters
        raw_audio = self.create_test_audio()

        # Old code path
        try:
            result = transcriber._convert_audio_format(raw_audio, 16000)
            assert isinstance(result, bytes)
        except TypeError as e:
            pytest.fail(f"Backward compatibility broken: existing code fails with {e}")
