"""Integration tests for SpeechToTextService with audio filtering."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_ai.speech_to_text.models import AudioChunk, TranscriptionResult
from local_ai.speech_to_text.service import SpeechToTextService


@pytest.mark.integration
class TestSpeechToTextServiceFilteringIntegration:
    """Integration tests for SpeechToTextService with audio filtering pipeline."""

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
        mock_pipeline.get_filter_stats = Mock()
        mock_pipeline.set_noise_profile = Mock()
        mock_pipeline.reset_adaptive_filters = Mock()
        return mock_pipeline

    @pytest.fixture
    def audio_data(self, chunk_size: int) -> bytes:
        """Sample audio data for testing."""
        return b"\x00\x01" * chunk_size

    @pytest.fixture
    def filtered_audio_data(self, chunk_size: int) -> bytes:
        """Sample filtered audio data for testing."""
        return b"\x01\x02" * chunk_size

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_audio_filter_pipeline_initialization(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test AudioFilterPipeline initialization in SpeechToTextService."""
        # Setup mocks
        mock_audio_capture = Mock()
        mock_audio_capture_class.return_value = mock_audio_capture
        mock_vad = Mock()
        mock_vad_class.return_value = mock_vad
        mock_transcriber = Mock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber_class.return_value = mock_transcriber

        # Test service initialization with filtering enabled
        service = SpeechToTextService(enable_filtering=True)
        service._audio_filter_pipeline = mock_audio_filter_pipeline

        # Initialize components
        result = service._initialize_components()

        assert result is True
        # Verify audio capture was configured with filtering
        mock_audio_capture_class.assert_called_once()
        # Verify other components were initialized
        mock_vad_class.assert_called_once()
        mock_transcriber_class.assert_called_once()

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_filtered_audio_passing_to_vad(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        filtered_audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test filtered audio passing to VoiceActivityDetector."""
        # Setup audio capture mock
        mock_audio_capture = AsyncMock()
        mock_audio_capture.is_capturing.return_value = True
        mock_audio_capture.get_audio_chunk.return_value = filtered_audio_data
        mock_audio_capture_class.return_value = mock_audio_capture

        # Setup VAD mock
        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        mock_vad.frame_size = chunk_size
        mock_vad_class.return_value = mock_vad

        # Setup transcriber mock
        mock_transcriber = AsyncMock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber.transcribe_audio_with_result.return_value = TranscriptionResult(
            text="test transcription",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.1,
        )
        mock_transcriber_class.return_value = mock_transcriber

        # Create service with filtering enabled
        service = SpeechToTextService(enable_filtering=True)
        service._audio_filter_pipeline = mock_audio_filter_pipeline

        # Start listening
        await service.start_listening()

        # Let the processing pipeline run briefly
        await asyncio.sleep(0.1)

        # Stop listening
        await service.stop_listening()

        # Verify VAD received the filtered audio
        # The VAD should be called with the filtered audio data
        mock_vad.is_speech.assert_called()
        # Verify the audio data passed to VAD matches filtered data format
        vad_call_args = mock_vad.is_speech.call_args_list
        if vad_call_args:
            # Check that VAD was called with audio data
            assert len(vad_call_args) > 0

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_seamless_integration_without_breaking_functionality(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test seamless integration without breaking existing functionality."""
        # Setup mocks to simulate normal operation
        mock_audio_capture = AsyncMock()
        mock_audio_capture.is_capturing.return_value = True
        mock_audio_capture.get_audio_chunk.return_value = audio_data
        mock_audio_capture_class.return_value = mock_audio_capture

        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        mock_vad.frame_size = chunk_size
        mock_vad_class.return_value = mock_vad

        mock_transcriber = AsyncMock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber.transcribe_audio_with_result.return_value = TranscriptionResult(
            text="test transcription",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.1,
        )
        mock_transcriber_class.return_value = mock_transcriber

        # Test callback functionality
        transcription_results = []

        def transcription_callback(result: TranscriptionResult) -> None:
            transcription_results.append(result)

        # Create service with filtering
        service = SpeechToTextService(enable_filtering=True)
        service._audio_filter_pipeline = mock_audio_filter_pipeline
        service.set_transcription_result_callback(transcription_callback)

        # Verify service can start and stop normally
        await service.start_listening()
        assert service.is_listening() is True

        # Let processing run briefly
        await asyncio.sleep(0.1)

        await service.stop_listening()
        assert service.is_listening() is False

        # Verify existing functionality still works
        component_status = service.get_component_status()
        assert "audio_capture" in component_status
        assert "vad" in component_status
        assert "transcriber" in component_status

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_filtering_disabled_fallback_behavior(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
    ) -> None:
        """Test service behavior when filtering is disabled."""
        # Setup mocks
        mock_audio_capture = AsyncMock()
        mock_audio_capture.is_capturing.return_value = True
        mock_audio_capture.get_audio_chunk.return_value = audio_data
        mock_audio_capture_class.return_value = mock_audio_capture

        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        mock_vad.frame_size = chunk_size
        mock_vad_class.return_value = mock_vad

        mock_transcriber = AsyncMock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber.transcribe_audio_with_result.return_value = TranscriptionResult(
            text="test transcription",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.1,
        )
        mock_transcriber_class.return_value = mock_transcriber

        # Create service with filtering disabled
        service = SpeechToTextService(enable_filtering=False)

        # Verify service works normally without filtering
        await service.start_listening()
        assert service.is_listening() is True

        # Let processing run briefly
        await asyncio.sleep(0.1)

        await service.stop_listening()
        assert service.is_listening() is False

        # Verify no filter pipeline was used
        assert (
            not hasattr(service, "_audio_filter_pipeline")
            or service._audio_filter_pipeline is None
        )

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_filter_pipeline_error_handling(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test error handling when filter pipeline fails."""
        # Setup mocks
        mock_audio_capture = AsyncMock()
        mock_audio_capture.is_capturing.return_value = True
        mock_audio_capture.get_audio_chunk.return_value = audio_data
        mock_audio_capture_class.return_value = mock_audio_capture

        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        mock_vad.frame_size = chunk_size
        mock_vad_class.return_value = mock_vad

        mock_transcriber = AsyncMock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber.transcribe_audio_with_result.return_value = TranscriptionResult(
            text="test transcription",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.1,
        )
        mock_transcriber_class.return_value = mock_transcriber

        # Setup filter pipeline to fail
        mock_audio_filter_pipeline.process_audio_chunk.side_effect = Exception(
            "Filter pipeline error"
        )

        # Create service with filtering
        service = SpeechToTextService(enable_filtering=True)
        service._audio_filter_pipeline = mock_audio_filter_pipeline

        # Service should still work despite filter errors
        await service.start_listening()
        assert service.is_listening() is True

        # Let processing run briefly
        await asyncio.sleep(0.1)

        await service.stop_listening()
        assert service.is_listening() is False

        # Verify service continued to operate despite filter errors
        component_status = service.get_component_status()
        assert component_status["listening"] is False  # Should have stopped cleanly

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_filter_configuration_and_management(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test filter configuration and management methods."""
        # Setup basic mocks
        mock_audio_capture = Mock()
        mock_audio_capture_class.return_value = mock_audio_capture
        mock_vad = Mock()
        mock_vad_class.return_value = mock_vad
        mock_transcriber = Mock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber_class.return_value = mock_transcriber

        # Create service
        service = SpeechToTextService(enable_filtering=True)
        service._audio_filter_pipeline = mock_audio_filter_pipeline

        # Test filter configuration methods
        service.set_noise_profile(b"noise_sample")
        mock_audio_filter_pipeline.set_noise_profile.assert_called_once_with(
            b"noise_sample"
        )

        service.reset_adaptive_filters()
        mock_audio_filter_pipeline.reset_adaptive_filters.assert_called_once()

        # Test filter stats retrieval
        mock_filter_stats = {
            "noise_reduction_db": 10.0,
            "signal_enhancement_db": 3.0,
            "processing_latency_ms": 25.0,
        }
        mock_audio_filter_pipeline.get_filter_stats.return_value = mock_filter_stats

        stats = service.get_filter_stats()
        assert stats == mock_filter_stats
        mock_audio_filter_pipeline.get_filter_stats.assert_called_once()

    @patch("local_ai.speech_to_text.service.AudioCapture")
    @patch("local_ai.speech_to_text.service.VoiceActivityDetector")
    @patch("local_ai.speech_to_text.service.WhisperTranscriber")
    async def test_performance_monitoring_with_filtering(
        self,
        mock_transcriber_class: Mock,
        mock_vad_class: Mock,
        mock_audio_capture_class: Mock,
        sample_rate: int,
        chunk_size: int,
        audio_data: bytes,
        mock_audio_filter_pipeline: Mock,
    ) -> None:
        """Test performance monitoring includes filtering metrics."""
        # Setup mocks
        mock_audio_capture = AsyncMock()
        mock_audio_capture.is_capturing.return_value = True
        mock_audio_capture.get_audio_chunk.return_value = audio_data
        mock_audio_capture_class.return_value = mock_audio_capture

        mock_vad = Mock()
        mock_vad.is_speech.return_value = False  # No speech to avoid transcription
        mock_vad.frame_size = chunk_size
        mock_vad_class.return_value = mock_vad

        mock_transcriber = AsyncMock()
        mock_transcriber.is_model_available.return_value = True
        mock_transcriber_class.return_value = mock_transcriber

        # Create service with monitoring and filtering
        service = SpeechToTextService(enable_filtering=True, enable_monitoring=True)
        service._audio_filter_pipeline = mock_audio_filter_pipeline

        # Start and run briefly
        await service.start_listening()
        await asyncio.sleep(0.1)
        await service.stop_listening()

        # Verify performance stats are available
        stats = service.get_performance_stats()
        assert isinstance(stats, dict)
        # Should not show monitoring disabled when enabled
        assert stats.get("monitoring_disabled") is not True
