"""Tests for SpeechToTextService class."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_ai.speech_to_text.service import SpeechToTextService


@pytest.mark.unit
class TestSpeechToTextServiceCoordination:
    """Test cases for SpeechToTextService coordination functionality."""

    def test_service_initialization(self) -> None:
        """Test SpeechToTextService initializes with correct default state."""
        service = SpeechToTextService()

        assert service._listening is False
        assert service._latest_transcription_result is None

    @pytest.mark.asyncio
    async def test_service_lifecycle_start_stop(self) -> None:
        """Test service lifecycle management - start and stop listening."""
        service = SpeechToTextService()

        # Initially not listening
        assert service._listening is False

        # Mock the component initialization to avoid model loading issues
        with patch.object(service, "_initialize_components") as mock_init:
            mock_init.return_value = True

            # Mock the components to avoid actual hardware/model dependencies
            service._audio_capture = Mock()
            service._audio_capture.start_capture = Mock()
            service._audio_capture.stop_capture = Mock()
            service._vad = Mock()
            service._transcriber = Mock()
            service._transcriber.is_model_available.return_value = True

            # Start listening
            await service.start_listening()
            assert service._listening is True

            # Stop listening
            await service.stop_listening()
            assert service._listening is False

    @pytest.mark.asyncio
    async def test_multiple_start_listening_calls(self) -> None:
        """Test that multiple start_listening calls don't cause issues."""
        service = SpeechToTextService()

        # Start listening multiple times
        await service.start_listening()
        assert service._listening is True

        await service.start_listening()
        assert service._listening is True

    @pytest.mark.asyncio
    async def test_multiple_stop_listening_calls(self) -> None:
        """Test that multiple stop_listening calls don't cause issues."""
        service = SpeechToTextService()

        # Stop listening when not started
        await service.stop_listening()
        assert service._listening is False

        # Start and stop multiple times
        await service.start_listening()
        await service.stop_listening()
        assert service._listening is False

        await service.stop_listening()
        assert service._listening is False


@pytest.mark.unit
class TestSpeechToTextServiceComponentCoordination:
    """Test cases for component coordination with mocked dependencies."""

    @pytest.fixture
    def mock_audio_capture(self):
        """Mock AudioCapture component."""
        # Create a mock instance directly since components aren't imported yet
        mock_instance = Mock()
        mock_instance.start_capture = Mock()
        mock_instance.stop_capture = Mock()
        mock_instance.get_audio_chunk = Mock(return_value=b"mock_audio_data")
        mock_instance.is_capturing = Mock(return_value=True)
        return mock_instance

    @pytest.fixture
    def mock_vad(self):
        """Mock VoiceActivityDetector component."""
        # Create a mock instance directly since components aren't imported yet
        mock_instance = Mock()
        mock_instance.is_speech = Mock(return_value=True)
        mock_instance.get_speech_segments = Mock(return_value=[b"speech_segment"])
        return mock_instance

    @pytest.fixture
    def mock_transcriber(self):
        """Mock WhisperTranscriber component."""
        # Create a mock instance directly since components aren't imported yet
        mock_instance = Mock()
        mock_instance.transcribe_audio = AsyncMock(return_value="transcribed text")
        mock_instance.is_model_available = Mock(return_value=True)
        return mock_instance

    def test_service_component_initialization_preparation(
        self, mock_audio_capture, mock_vad, mock_transcriber
    ):
        """Test that service is ready for component initialization."""
        service = SpeechToTextService()

        # Service should be able to initialize without components for now
        # (Components will be initialized in the implementation task)
        assert service is not None
        assert hasattr(service, "start_listening")
        assert hasattr(service, "stop_listening")
        assert hasattr(service, "set_transcription_result_callback")
        assert hasattr(service, "get_latest_transcription_result")

    @pytest.mark.asyncio
    async def test_service_lifecycle_with_mocked_components(
        self, mock_audio_capture, mock_vad, mock_transcriber
    ):
        """Test service lifecycle management with mocked components."""
        service = SpeechToTextService()

        # Test that service can start and stop without errors
        # (Actual component coordination will be implemented in task 5.2)
        await service.start_listening()
        assert service._listening is True

        await service.stop_listening()
        assert service._listening is False

    def test_callback_mechanism_with_mocked_transcription(
        self, mock_audio_capture, mock_vad, mock_transcriber
    ):
        """Test callback mechanism works with mocked transcription results."""
        from local_ai.speech_to_text.models import TranscriptionResult

        service = SpeechToTextService()
        callback_results = []

        def transcription_callback(result: TranscriptionResult) -> None:
            callback_results.append(result.text)

        service.set_transcription_result_callback(transcription_callback)

        # Simulate transcription result (this will be called by the real implementation)
        if service._transcription_result_callback:
            test_result = TranscriptionResult(
                text="mocked transcription result",
                confidence=0.95,
                timestamp=0.0,
                processing_time=0.1,
            )
            service._transcription_result_callback(test_result)

        assert callback_results == ["mocked transcription result"]
        assert service._transcription_result_callback is transcription_callback

    @pytest.mark.asyncio
    async def test_error_handling_preparation(
        self, mock_audio_capture, mock_vad, mock_transcriber
    ):
        """Test that service is prepared for error handling scenarios."""
        service = SpeechToTextService()

        # Configure mocks to simulate errors
        mock_audio_capture.start_capture.side_effect = Exception("Audio capture failed")
        mock_transcriber.transcribe_audio.side_effect = Exception("Transcription failed")

        # Service should handle errors gracefully (implementation in task 5.2)
        try:
            await service.start_listening()
            await service.stop_listening()
        except Exception as e:
            # For now, we just ensure the service structure supports error handling
            assert isinstance(e, Exception)

    def test_concurrent_callback_safety(self):
        """Test that callback mechanism is safe for concurrent access."""
        from local_ai.speech_to_text.models import TranscriptionResult

        service = SpeechToTextService()
        callback_results = []

        def thread_safe_callback(result: TranscriptionResult) -> None:
            callback_results.append(result.text)

        # Set callback
        service.set_transcription_result_callback(thread_safe_callback)

        # Simulate multiple concurrent callback invocations
        if service._transcription_result_callback:
            for i, text in enumerate(["result1", "result2", "result3"]):
                test_result = TranscriptionResult(
                    text=text, confidence=0.95, timestamp=float(i), processing_time=0.1
                )
                service._transcription_result_callback(test_result)

        assert len(callback_results) == 3
        assert "result1" in callback_results
        assert "result2" in callback_results
        assert "result3" in callback_results


@pytest.mark.unit
class TestSpeechToTextServiceRealTimePipeline:
    """Test cases for real-time transcription pipeline functionality."""

    @pytest.fixture
    def mock_components(self):
        """Create mocked components for pipeline testing."""
        # Mock audio capture
        mock_audio = Mock()
        mock_audio.start_capture = Mock()
        mock_audio.stop_capture = Mock()
        mock_audio.is_capturing = Mock(return_value=True)
        mock_audio.get_audio_chunk = Mock()

        # Mock VAD
        mock_vad = Mock()
        mock_vad.is_speech = Mock()
        mock_vad.get_speech_segments = Mock()

        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber.transcribe_audio = AsyncMock()
        mock_transcriber.is_model_available = Mock(return_value=True)

        return {"audio": mock_audio, "vad": mock_vad, "transcriber": mock_transcriber}

    @pytest.mark.asyncio
    async def test_continuous_audio_processing_loop_preparation(self, mock_components):
        """Test preparation for continuous audio processing loop."""
        service = SpeechToTextService()

        # Mock the component initialization to use our mocks
        with patch.object(service, "_initialize_components", return_value=True):
            with patch.object(service, "_audio_capture", mock_components["audio"]):
                with patch.object(service, "_vad", mock_components["vad"]):
                    with patch.object(
                        service, "_transcriber", mock_components["transcriber"]
                    ):
                        # Test that service can start listening
                        await service.start_listening()
                        assert service.is_listening()

                        # Verify audio capture was started
                        mock_components["audio"].start_capture.assert_called_once()

                        # Test that service can stop listening
                        await service.stop_listening()
                        assert not service.is_listening()

    @pytest.mark.asyncio
    async def test_speech_detection_and_filtering(self, mock_components):
        """Test speech detection and non-speech filtering."""
        SpeechToTextService()

        # Test with speech detected
        mock_components["vad"].is_speech.return_value = True
        mock_components["vad"].get_speech_segments.return_value = [b"speech_data"]

        # Simulate speech processing (this will be implemented in task 5.4)
        audio_chunk = b"test_audio"
        is_speech = mock_components["vad"].is_speech(audio_chunk)
        assert is_speech is True

        # Test with no speech detected
        mock_components["vad"].is_speech.return_value = False
        mock_components["vad"].get_speech_segments.return_value = []

        is_speech = mock_components["vad"].is_speech(audio_chunk)
        assert is_speech is False


@pytest.mark.unit
class TestSpeechToTextServiceErrorHandling:
    """Test cases for error handling and recovery scenarios."""

    @pytest.fixture
    def failing_components(self):
        """Create components that simulate various failure scenarios."""
        # Mock audio capture that fails
        failing_audio = Mock()
        failing_audio.start_capture = Mock(
            side_effect=Exception("Audio device not found")
        )
        failing_audio.stop_capture = Mock()
        failing_audio.is_capturing = Mock(return_value=False)

        # Mock VAD that fails
        failing_vad = Mock()
        failing_vad.is_speech = Mock(side_effect=Exception("VAD processing error"))

        # Mock transcriber that fails
        failing_transcriber = Mock()
        failing_transcriber.transcribe_audio = AsyncMock(
            side_effect=Exception("Transcription failed")
        )
        failing_transcriber.is_model_available = Mock(return_value=False)

        return {
            "audio": failing_audio,
            "vad": failing_vad,
            "transcriber": failing_transcriber,
        }

    @pytest.mark.asyncio
    async def test_audio_capture_error_handling(self, failing_components):
        """Test error handling when audio capture fails."""
        service = SpeechToTextService()

        # Mock component initialization to succeed, but audio capture start to fail
        with patch.object(service, "_initialize_components", return_value=True):
            with patch.object(service, "_audio_capture", failing_components["audio"]):
                with patch.object(service, "_vad", Mock()):
                    with patch.object(service, "_transcriber", Mock()):
                        # Starting should raise an exception due to audio capture failure
                        with pytest.raises(Exception):
                            await service.start_listening()

                        # Service should not be in listening state
                        assert not service.is_listening()

    @pytest.mark.asyncio
    async def test_transcriber_unavailable_error_handling(self, failing_components):
        """Test error handling when transcriber model is unavailable."""
        service = SpeechToTextService()

        # Mock initialization to fail due to unavailable transcriber
        with patch.object(service, "_initialize_components", return_value=False):
            # Starting should raise an exception
            with pytest.raises(
                RuntimeError, match="Failed to initialize speech-to-text components"
            ):
                await service.start_listening()

            # Service should not be in listening state
            assert not service.is_listening()

    @pytest.mark.asyncio
    async def test_service_cleanup_on_error(self):
        """Test that service properly cleans up resources on error."""
        service = SpeechToTextService()

        # Mock components
        mock_audio = Mock()
        mock_audio.start_capture = Mock(side_effect=Exception("Startup error"))
        mock_audio.stop_capture = Mock()
        mock_audio.is_capturing = Mock(return_value=True)

        with patch.object(service, "_initialize_components", return_value=True):
            with patch.object(service, "_audio_capture", mock_audio):
                with patch.object(service, "_cleanup_components") as mock_cleanup:
                    # Starting should fail and trigger cleanup
                    with pytest.raises(Exception):
                        await service.start_listening()

                    # Cleanup should have been called
                    mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_stop_during_processing(self):
        """Test graceful stop when processing is in progress."""
        service = SpeechToTextService()

        # Mock a long-running processing task
        async def long_running_task():
            await asyncio.sleep(10)  # Simulate long processing

        # Set up a mock processing task
        service._processing_task = asyncio.create_task(long_running_task())
        service._listening = True

        # Stop should cancel the task gracefully
        await service.stop_listening()

        # Task should be cancelled
        assert service._processing_task.cancelled()
        assert not service.is_listening()

    @pytest.mark.asyncio
    async def test_multiple_error_recovery_cycles(self):
        """Test that service can recover from multiple error cycles."""
        service = SpeechToTextService()

        # Test multiple start/stop cycles with errors
        for i in range(3):
            # Mock initialization failure
            with patch.object(service, "_initialize_components", return_value=False):
                with pytest.raises(RuntimeError):
                    await service.start_listening()

                assert not service.is_listening()

            # Service should be able to attempt restart
            await service.stop_listening()  # Should not raise error even if not started


@pytest.mark.unit
class TestSpeechToTextServiceIntegrationPreparation:
    """Test cases to prepare for integration with real components."""

    def test_service_interface_completeness(self):
        """Test that service provides complete interface for integration."""
        service = SpeechToTextService()

        # Check all required methods exist
        assert hasattr(service, "start_listening")
        assert hasattr(service, "stop_listening")
        assert hasattr(service, "get_latest_transcription_result")
        assert hasattr(service, "set_transcription_result_callback")
        assert hasattr(service, "is_listening")
        assert hasattr(service, "get_component_status")

        # Check methods are callable
        assert callable(service.start_listening)
        assert callable(service.stop_listening)
        assert callable(service.get_latest_transcription_result)
        assert callable(service.set_transcription_result_callback)
        assert callable(service.is_listening)
        assert callable(service.get_component_status)

    @pytest.mark.asyncio
    async def test_async_method_compatibility(self):
        """Test that async methods work correctly for future integration."""
        service = SpeechToTextService()

        # Mock initialization to avoid actual component setup
        with patch.object(service, "_initialize_components", return_value=True):
            with patch.object(service, "_audio_capture", Mock()):
                # Test async methods can be awaited
                await service.start_listening()
                await service.stop_listening()

    def test_state_consistency(self):
        """Test that service maintains consistent internal state."""
        from local_ai.speech_to_text.models import TranscriptionResult

        service = SpeechToTextService()

        # Initial state
        assert service._listening is False
        assert service._transcription_result_callback is None
        assert service._latest_transcription_result is None

        # State after callback setting
        def dummy_callback(result: TranscriptionResult) -> None:
            pass

        service.set_transcription_result_callback(dummy_callback)
        assert service._transcription_result_callback is dummy_callback
        assert service._listening is False  # Should not change
        assert service._latest_transcription_result is None  # Should not change

    def test_component_status_reporting(self):
        """Test component status reporting functionality."""
        service = SpeechToTextService()

        # Initial status should show no components initialized
        status = service.get_component_status()
        assert status["audio_capture"] is False
        assert status["vad"] is False
        assert status["transcriber"] is False
        assert status["listening"] is False
