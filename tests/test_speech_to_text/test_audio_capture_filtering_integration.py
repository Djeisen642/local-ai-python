"""Integration tests for AudioCapture modifications with audio filtering."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_ai.speech_to_text.audio_capture import AudioCapture
from local_ai.speech_to_text.models import AudioChunk


@pytest.mark.integration
class TestAudioCaptureFilteringIntegration:
    """Integration tests for AudioCapture with audio filtering pipeline."""

    @pytest.fixture
    def sample_rate(self) -> int:
        """Sample rate for testing."""
        return 16000

    @pytest.fixture
    def chunk_size(self) -> int:
        """Chunk size for testing."""
        return 1024

    @pytest.fixture
    def mock_audio_filter_pipeline(self) -> Mock:
        """Mock AudioFilterPipeline for testing."""
        mock_pipeline = AsyncMock()
        mock_pipeline.process_audio_chunk = AsyncMock()
        return mock_pipeline

    @pytest.fixture
    def audio_data(self, chunk_size: int) -> bytes:
        """Sample audio data for testing."""
        return b"\x00\x01" * chunk_size

    @pytest.fixture
    def filtered_audio_data(self, chunk_size: int) -> bytes:
        """Sample filtered audio data for testing."""
        return b"\x01\x02" * chunk_size

    @patch("pyaudio.PyAudio")
    async def test_audio_capture_with_filtering_enabled(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        filtered_audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test AudioCapture.get_audio_chunk() integration with filtering pipeline."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = audio_data

        # Setup filter pipeline mock
        filtered_chunk = AudioChunk(
            data=filtered_audio_data,
            timestamp=1234567890.0,
            sample_rate=sample_rate,
            duration=chunk_size / sample_rate,
            is_filtered=True,
        )
        mock_audio_filter_pipeline.process_audio_chunk.return_value = filtered_chunk

        # Test with filtering enabled
        capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size)
        capture._audio_filter = mock_audio_filter_pipeline  # Inject mock filter
        capture._enable_filtering = True

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Verify filtering was applied
        assert result == filtered_audio_data
        mock_audio_filter_pipeline.process_audio_chunk.assert_called_once()

        # Verify the input to the filter was correct
        call_args = mock_audio_filter_pipeline.process_audio_chunk.call_args[0][0]
        assert isinstance(call_args, AudioChunk)
        assert call_args.data == audio_data
        assert call_args.sample_rate == sample_rate

        capture.stop_capture()

    @patch("pyaudio.PyAudio")
    async def test_audio_capture_with_filtering_disabled(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test AudioCapture passes through unfiltered audio when filtering disabled."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = audio_data

        # Test with filtering disabled
        capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size)
        capture._audio_filter = mock_audio_filter_pipeline  # Inject mock filter
        capture._enable_filtering = False

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Verify no filtering was applied
        assert result == audio_data
        mock_audio_filter_pipeline.process_audio_chunk.assert_not_called()

        capture.stop_capture()

    @patch("pyaudio.PyAudio")
    async def test_audio_format_compatibility_with_filtering(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test audio format and chunk size compatibility with filtering pipeline."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = audio_data

        # Setup filter pipeline to return compatible format
        filtered_chunk = AudioChunk(
            data=audio_data,  # Same format as input
            timestamp=1234567890.0,
            sample_rate=sample_rate,
            duration=chunk_size / sample_rate,
            is_filtered=True,
        )
        mock_audio_filter_pipeline.process_audio_chunk.return_value = filtered_chunk

        capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size)
        capture._audio_filter = mock_audio_filter_pipeline
        capture._enable_filtering = True

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Verify format compatibility
        assert isinstance(result, bytes)
        assert len(result) == len(audio_data)

        # Verify AudioChunk passed to filter has correct format
        call_args = mock_audio_filter_pipeline.process_audio_chunk.call_args[0][0]
        assert call_args.sample_rate == sample_rate
        assert call_args.duration == chunk_size / sample_rate
        assert len(call_args.data) == len(audio_data)

        capture.stop_capture()

    @patch("pyaudio.PyAudio")
    async def test_configuration_option_enable_disable_filtering(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test configuration option to enable/disable filtering."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = audio_data

        filtered_chunk = AudioChunk(
            data=b"\x02\x03" * chunk_size,
            timestamp=1234567890.0,
            sample_rate=sample_rate,
            duration=chunk_size / sample_rate,
            is_filtered=True,
        )
        mock_audio_filter_pipeline.process_audio_chunk.return_value = filtered_chunk

        # Test enabling filtering via configuration
        capture = AudioCapture(
            sample_rate=sample_rate, chunk_size=chunk_size, enable_filtering=True
        )
        capture._audio_filter = mock_audio_filter_pipeline

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Should use filtered audio
        assert result == filtered_chunk.data
        mock_audio_filter_pipeline.process_audio_chunk.assert_called_once()

        capture.stop_capture()
        mock_audio_filter_pipeline.reset_mock()

        # Test disabling filtering via configuration
        capture = AudioCapture(
            sample_rate=sample_rate, chunk_size=chunk_size, enable_filtering=False
        )
        capture._audio_filter = mock_audio_filter_pipeline

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Should use unfiltered audio
        assert result == audio_data
        mock_audio_filter_pipeline.process_audio_chunk.assert_not_called()

        capture.stop_capture()

    @patch("pyaudio.PyAudio")
    async def test_filtering_error_handling_graceful_fallback(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test graceful fallback to unfiltered audio when filtering fails."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = audio_data

        # Setup filter pipeline to raise an exception
        mock_audio_filter_pipeline.process_audio_chunk.side_effect = Exception(
            "Filter processing failed"
        )

        capture = AudioCapture(
            sample_rate=sample_rate, chunk_size=chunk_size, enable_filtering=True
        )
        capture._audio_filter = mock_audio_filter_pipeline

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Should fallback to unfiltered audio
        assert result == audio_data
        mock_audio_filter_pipeline.process_audio_chunk.assert_called_once()

        capture.stop_capture()

    @patch("pyaudio.PyAudio")
    async def test_filtering_with_multiple_chunks_consistency(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test filtering consistency across multiple audio chunks."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream

        # Create multiple different audio chunks
        audio_chunks = [
            b"\x00\x01" * chunk_size,
            b"\x01\x02" * chunk_size,
            b"\x02\x03" * chunk_size,
        ]
        mock_stream.read.side_effect = audio_chunks

        # Setup filter pipeline to return processed chunks
        filtered_chunks = [
            AudioChunk(
                data=b"\x10\x11" * chunk_size,
                timestamp=1234567890.0 + i,
                sample_rate=sample_rate,
                duration=chunk_size / sample_rate,
                is_filtered=True,
            )
            for i in range(3)
        ]
        mock_audio_filter_pipeline.process_audio_chunk.side_effect = filtered_chunks

        capture = AudioCapture(
            sample_rate=sample_rate, chunk_size=chunk_size, enable_filtering=True
        )
        capture._audio_filter = mock_audio_filter_pipeline

        capture.start_capture()

        # Process multiple chunks
        results = []
        for _ in range(3):
            result = await capture.get_audio_chunk()
            results.append(result)

        # Verify all chunks were processed through filter
        assert len(results) == 3
        assert mock_audio_filter_pipeline.process_audio_chunk.call_count == 3

        # Verify each result matches the filtered output
        for i, result in enumerate(results):
            assert result == filtered_chunks[i].data

        capture.stop_capture()

    @patch("pyaudio.PyAudio")
    async def test_filtering_preserves_audio_metadata(
        self,
        mock_pyaudio: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test that filtering preserves important audio metadata."""
        # Setup PyAudio mocks
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_stream = Mock()
        mock_pa_instance.open.return_value = mock_stream
        mock_stream.read.return_value = audio_data

        # Setup filter pipeline with metadata preservation
        filtered_chunk = AudioChunk(
            data=audio_data,
            timestamp=1234567890.0,
            sample_rate=sample_rate,
            duration=chunk_size / sample_rate,
            noise_level=0.1,
            signal_level=0.8,
            snr_db=15.0,
            is_filtered=True,
        )
        mock_audio_filter_pipeline.process_audio_chunk.return_value = filtered_chunk

        capture = AudioCapture(
            sample_rate=sample_rate, chunk_size=chunk_size, enable_filtering=True
        )
        capture._audio_filter = mock_audio_filter_pipeline

        capture.start_capture()
        result = await capture.get_audio_chunk()

        # Verify the input AudioChunk has correct metadata
        call_args = mock_audio_filter_pipeline.process_audio_chunk.call_args[0][0]
        assert call_args.sample_rate == sample_rate
        assert call_args.duration == chunk_size / sample_rate
        assert call_args.is_filtered is False  # Input should not be marked as filtered

        # Verify result is the filtered data
        assert result == filtered_chunk.data

        capture.stop_capture()
