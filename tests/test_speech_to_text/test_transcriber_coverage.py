"""Additional tests to improve transcriber.py coverage."""

from unittest.mock import MagicMock, patch

import pytest

from local_ai.speech_to_text.transcriber import WhisperTranscriber


@pytest.mark.unit
class TestWhisperTranscriberCoverage:
    """Test cases to improve coverage of WhisperTranscriber."""

    @pytest.fixture
    def transcriber(self) -> WhisperTranscriber:
        """Create a WhisperTranscriber instance for testing."""
        return WhisperTranscriber()

    def test_get_model_info_with_additional_attributes(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test get_model_info with model that has additional attributes."""
        mock_model = MagicMock()
        mock_model.model_size = "small"
        mock_model.device = "cpu"
        mock_model.compute_type = "int8"

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                info = transcriber.get_model_info()

        assert info["model_size"] == "small"
        assert info["actual_model_size"] == "small"
        assert info["device"] == "cpu"
        assert info["compute_type"] == "int8"

    def test_get_model_info_with_missing_attributes(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test get_model_info when model lacks some attributes."""
        mock_model = MagicMock()

        # Configure hasattr to return False for missing attributes
        def mock_hasattr(obj, attr):
            return attr not in ["model_size", "device", "compute_type"]

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                with patch("builtins.hasattr", side_effect=mock_hasattr):
                    transcriber._model_loaded = True  # Set loaded state
                    info = transcriber.get_model_info()

        assert info["model_size"] == "small"  # From constructor
        assert info["model_loaded"] is True
        # Should not have the missing attributes
        assert "actual_model_size" not in info
        assert "device" not in info
        assert "compute_type" not in info

    def test_get_model_info_exception_handling(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test get_model_info when the method itself raises exception."""
        # Make is_model_available return False to trigger the early return
        with patch.object(transcriber, "is_model_available", return_value=False):
            info = transcriber.get_model_info()

        # Should return empty dict when model not available
        assert info == {}

    def test_create_minimal_wav(self, transcriber: WhisperTranscriber) -> None:
        """Test _create_minimal_wav method."""
        minimal_wav = transcriber._create_minimal_wav()

        assert isinstance(minimal_wav, bytes)
        assert len(minimal_wav) > 44  # Should have WAV header + some data
        assert minimal_wav.startswith(b"RIFF")
        assert b"WAVE" in minimal_wav

    def test_resample_audio_same_rate(self, transcriber: WhisperTranscriber) -> None:
        """Test _resample_audio when source and target rates are the same."""
        import numpy as np

        samples = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        resampled = transcriber._resample_audio(samples, 16000, 16000)

        # Should return original samples unchanged
        np.testing.assert_array_equal(resampled, samples)

    def test_resample_audio_different_rates(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test _resample_audio with different source and target rates."""
        import numpy as np

        samples = np.array([1, 2, 3, 4], dtype=np.int16)
        resampled = transcriber._resample_audio(samples, 8000, 16000)

        # Should have approximately double the length (upsampling)
        assert len(resampled) == 8  # 4 * (16000/8000) = 8
        assert resampled.dtype == np.int16

    def test_create_wav_data_mono_audio(self, transcriber: WhisperTranscriber) -> None:
        """Test _create_wav_data with mono audio."""
        import numpy as np

        # Create test audio data (16-bit mono)
        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        audio_bytes = samples.tobytes()

        wav_data = transcriber._create_wav_data(audio_bytes, 16000, 1, 16000)

        assert isinstance(wav_data, bytes)
        assert wav_data.startswith(b"RIFF")
        assert b"WAVE" in wav_data

    def test_create_wav_data_stereo_to_mono(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test _create_wav_data converting stereo to mono."""
        import numpy as np

        # Create stereo audio data (left and right channels)
        left_channel = np.array([100, 300], dtype=np.int16)
        right_channel = np.array([200, 400], dtype=np.int16)
        stereo_samples = np.column_stack((left_channel, right_channel)).flatten()
        audio_bytes = stereo_samples.tobytes()

        wav_data = transcriber._create_wav_data(audio_bytes, 16000, 2, 16000)

        assert isinstance(wav_data, bytes)
        assert wav_data.startswith(b"RIFF")
        assert b"WAVE" in wav_data

    def test_create_wav_data_with_resampling(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test _create_wav_data with resampling."""
        import numpy as np

        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        audio_bytes = samples.tobytes()

        # Resample from 8kHz to 16kHz
        wav_data = transcriber._create_wav_data(audio_bytes, 8000, 1, 16000)

        assert isinstance(wav_data, bytes)
        assert wav_data.startswith(b"RIFF")
        assert b"WAVE" in wav_data

    def test_create_wav_data_exception_handling(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test _create_wav_data exception handling."""
        # Pass invalid audio data
        invalid_audio = b"invalid"

        wav_data = transcriber._create_wav_data(invalid_audio, 16000, 1, 16000)

        # Should return minimal WAV on error
        assert isinstance(wav_data, bytes)
        assert wav_data.startswith(b"RIFF")

    def test_post_process_text_edge_cases(self, transcriber: WhisperTranscriber) -> None:
        """Test _post_process_text with various edge cases."""
        # Test None input
        result = transcriber._post_process_text(None)
        assert result == ""

        # Test non-string input
        result = transcriber._post_process_text(123)
        assert result == ""

        # Test excessive whitespace
        result = transcriber._post_process_text("  hello    world  \n\t  ")
        assert result == "Hello world"

        # Test already capitalized
        result = transcriber._post_process_text("Hello world")
        assert result == "Hello world"

        # Test empty after strip
        result = transcriber._post_process_text("   \n\t   ")
        assert result == ""

    def test_convert_audio_format_edge_cases(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test _convert_audio_format with edge cases."""
        # Test None input
        result = transcriber._convert_audio_format(None)
        assert result == b""

        # Test non-bytes input
        result = transcriber._convert_audio_format("not bytes")
        assert result == b""

        # Test empty bytes
        result = transcriber._convert_audio_format(b"")
        assert result == b""

        # Test invalid WAV data
        invalid_wav = b"RIFF" + b"x" * 100 + b"WAVE" + b"invalid"
        result = transcriber._convert_audio_format(invalid_wav)
        # Should return original data if parsing fails
        assert result == invalid_wav

    @pytest.mark.asyncio
    async def test_transcribe_audio_edge_cases(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio with various edge cases."""
        # Test None input
        result = await transcriber.transcribe_audio(None)
        assert result == ""

        # Test non-bytes input
        result = await transcriber.transcribe_audio("not bytes")
        assert result == ""

        # Test empty bytes
        result = await transcriber.transcribe_audio(b"")
        assert result == ""

    def test_load_model_import_error(self, transcriber: WhisperTranscriber) -> None:
        """Test _load_model when faster_whisper is not available."""
        with patch("local_ai.speech_to_text.transcriber.faster_whisper", None):
            result = transcriber._load_model()
            assert result is False

    def test_load_model_file_not_found(self, transcriber: WhisperTranscriber) -> None:
        """Test _load_model when model files are not found."""
        with patch("local_ai.speech_to_text.transcriber.faster_whisper") as mock_fw:
            mock_fw.WhisperModel.side_effect = FileNotFoundError("Model not found")

            result = transcriber._load_model()
            assert result is False

    def test_load_model_general_exception(self, transcriber: WhisperTranscriber) -> None:
        """Test _load_model with general exception."""
        with patch("local_ai.speech_to_text.transcriber.faster_whisper") as mock_fw:
            mock_fw.WhisperModel.side_effect = Exception("General error")

            result = transcriber._load_model()
            assert result is False

    def test_is_model_available_exception(self, transcriber: WhisperTranscriber) -> None:
        """Test is_model_available when _load_model raises exception."""
        with patch.object(
            transcriber, "_load_model", side_effect=Exception("Load error")
        ):
            result = transcriber.is_model_available()
            assert result is False
