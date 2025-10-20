"""Tests for WhisperTranscriber transcribe_audio_with_result method."""

from unittest.mock import MagicMock, patch

import pytest

from local_ai.speech_to_text.models import TranscriptionResult
from local_ai.speech_to_text.transcriber import WhisperTranscriber


@pytest.mark.unit
class TestWhisperTranscriberWithResult:
    """Test cases for transcribe_audio_with_result method."""

    @pytest.fixture
    def transcriber(self) -> WhisperTranscriber:
        """Create a WhisperTranscriber instance for testing."""
        return WhisperTranscriber()

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_success(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test successful transcription with result details."""
        # Mock the model and its methods
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.avg_logprob = -0.5  # Mock confidence score
        mock_segment.start = 0.0  # Add timing info for confidence calculation
        mock_segment.end = 2.0

        mock_model.transcribe.return_value = ([mock_segment], {})

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "test.wav"
                    mock_temp.return_value.__enter__.return_value.write = MagicMock()
                    mock_temp.return_value.__enter__.return_value.flush = MagicMock()

                    with patch("os.unlink"):
                        result = await transcriber.transcribe_audio_with_result(
                            b"fake_audio_data"
                        )

        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.confidence > 0.0
        assert result.processing_time > 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_empty_input(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result with empty input."""
        result = await transcriber.transcribe_audio_with_result(b"")

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_invalid_input(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result with invalid input."""
        result = await transcriber.transcribe_audio_with_result(None)

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_large_audio(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result with excessively large audio."""
        large_audio = b"x" * (101 * 1024 * 1024)  # 101MB
        result = await transcriber.transcribe_audio_with_result(large_audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_model_unavailable(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result when model is unavailable."""
        with patch.object(transcriber, "_load_model", return_value=False):
            result = await transcriber.transcribe_audio_with_result(b"fake_audio_data")

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_conversion_failure(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result when audio conversion fails."""
        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_convert_audio_format", return_value=b""):
                result = await transcriber.transcribe_audio_with_result(
                    b"fake_audio_data"
                )

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_timeout_error(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result with timeout error."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = TimeoutError("Transcription timeout")

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "test.wav"
                    mock_temp.return_value.__enter__.return_value.write = MagicMock()
                    mock_temp.return_value.__enter__.return_value.flush = MagicMock()

                    with patch("os.unlink"):
                        result = await transcriber.transcribe_audio_with_result(
                            b"fake_audio_data"
                        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_memory_error(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result with memory error."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = MemoryError("Out of memory")

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "test.wav"
                    mock_temp.return_value.__enter__.return_value.write = MagicMock()
                    mock_temp.return_value.__enter__.return_value.flush = MagicMock()

                    with patch("os.unlink"):
                        result = await transcriber.transcribe_audio_with_result(
                            b"fake_audio_data"
                        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_multiple_segments(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result with multiple segments."""
        mock_model = MagicMock()

        # Create multiple mock segments
        mock_segment1 = MagicMock()
        mock_segment1.text = "Hello "
        mock_segment1.avg_logprob = -0.3
        mock_segment1.start = 0.0  # Add timing info for confidence calculation
        mock_segment1.end = 1.0

        mock_segment2 = MagicMock()
        mock_segment2.text = "world"
        mock_segment2.avg_logprob = -0.7
        mock_segment2.start = 1.0  # Add timing info for confidence calculation
        mock_segment2.end = 2.0

        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], {})

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "test.wav"
                    mock_temp.return_value.__enter__.return_value.write = MagicMock()
                    mock_temp.return_value.__enter__.return_value.flush = MagicMock()

                    with patch("os.unlink"):
                        result = await transcriber.transcribe_audio_with_result(
                            b"fake_audio_data"
                        )

        # Verify result combines segments
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.confidence > 0.0  # Should be average of segment confidences

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_result_no_confidence_data(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Test transcribe_audio_with_result when segments have no confidence data."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        # No avg_logprob attribute
        del mock_segment.avg_logprob

        mock_model.transcribe.return_value = ([mock_segment], {})

        with patch.object(transcriber, "_load_model", return_value=True):
            with patch.object(transcriber, "_model", mock_model):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "test.wav"
                    mock_temp.return_value.__enter__.return_value.write = MagicMock()
                    mock_temp.return_value.__enter__.return_value.flush = MagicMock()

                    with patch("os.unlink"):
                        result = await transcriber.transcribe_audio_with_result(
                            b"fake_audio_data"
                        )

        # Should handle missing confidence gracefully
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.confidence == 0.0  # Default when no confidence available
