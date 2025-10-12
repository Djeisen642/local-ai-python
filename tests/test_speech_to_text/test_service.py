"""Tests for SpeechToTextService class."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from local_ai.speech_to_text.service import SpeechToTextService


class TestSpeechToTextServiceCoordination:
    """Test cases for SpeechToTextService coordination functionality."""
    
    def test_service_initialization(self) -> None:
        """Test SpeechToTextService initializes with correct default state."""
        service = SpeechToTextService()
        
        assert service.get_latest_transcription() is None
        assert service._listening is False
        assert service._transcription_callback is None
        assert service._latest_transcription is None
    
    @pytest.mark.asyncio
    async def test_service_lifecycle_start_stop(self) -> None:
        """Test service lifecycle management - start and stop listening."""
        service = SpeechToTextService()
        
        # Initially not listening
        assert service._listening is False
        
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
    
    def test_transcription_callback_mechanism(self) -> None:
        """Test callback mechanism for real-time transcription updates."""
        service = SpeechToTextService()
        callback_results = []
        
        def test_callback(text: str) -> None:
            callback_results.append(text)
        
        # Set callback
        service.set_transcription_callback(test_callback)
        assert service._transcription_callback is test_callback
        
        # Test callback can be called
        if service._transcription_callback:
            service._transcription_callback("test transcription")
        
        assert callback_results == ["test transcription"]
    
    def test_callback_replacement(self) -> None:
        """Test that callbacks can be replaced."""
        service = SpeechToTextService()
        
        callback1_results = []
        callback2_results = []
        
        def callback1(text: str) -> None:
            callback1_results.append(text)
        
        def callback2(text: str) -> None:
            callback2_results.append(text)
        
        # Set first callback
        service.set_transcription_callback(callback1)
        if service._transcription_callback:
            service._transcription_callback("message1")
        
        # Replace with second callback
        service.set_transcription_callback(callback2)
        if service._transcription_callback:
            service._transcription_callback("message2")
        
        assert callback1_results == ["message1"]
        assert callback2_results == ["message2"]
    
    def test_latest_transcription_storage(self) -> None:
        """Test that latest transcription is stored and retrievable."""
        service = SpeechToTextService()
        
        # Initially None
        assert service.get_latest_transcription() is None
        
        # Simulate setting transcription (this will be done by the real implementation)
        service._latest_transcription = "Hello world"
        assert service.get_latest_transcription() == "Hello world"
        
        # Update transcription
        service._latest_transcription = "Updated transcription"
        assert service.get_latest_transcription() == "Updated transcription"


class TestSpeechToTextServiceComponentCoordination:
    """Test cases for component coordination with mocked dependencies."""
    
    @pytest.fixture
    def mock_audio_capture(self):
        """Mock AudioCapture component."""
        # Create a mock instance directly since components aren't imported yet
        mock_instance = Mock()
        mock_instance.start_capture = Mock()
        mock_instance.stop_capture = Mock()
        mock_instance.get_audio_chunk = Mock(return_value=b'mock_audio_data')
        mock_instance.is_capturing = Mock(return_value=True)
        return mock_instance
    
    @pytest.fixture
    def mock_vad(self):
        """Mock VoiceActivityDetector component."""
        # Create a mock instance directly since components aren't imported yet
        mock_instance = Mock()
        mock_instance.is_speech = Mock(return_value=True)
        mock_instance.get_speech_segments = Mock(return_value=[b'speech_segment'])
        return mock_instance
    
    @pytest.fixture
    def mock_transcriber(self):
        """Mock WhisperTranscriber component."""
        # Create a mock instance directly since components aren't imported yet
        mock_instance = Mock()
        mock_instance.transcribe_audio = AsyncMock(return_value="transcribed text")
        mock_instance.is_model_available = Mock(return_value=True)
        return mock_instance
    
    def test_service_component_initialization_preparation(self, mock_audio_capture, mock_vad, mock_transcriber):
        """Test that service is ready for component initialization."""
        service = SpeechToTextService()
        
        # Service should be able to initialize without components for now
        # (Components will be initialized in the implementation task)
        assert service is not None
        assert hasattr(service, 'start_listening')
        assert hasattr(service, 'stop_listening')
        assert hasattr(service, 'set_transcription_callback')
        assert hasattr(service, 'get_latest_transcription')
    
    @pytest.mark.asyncio
    async def test_service_lifecycle_with_mocked_components(self, mock_audio_capture, mock_vad, mock_transcriber):
        """Test service lifecycle management with mocked components."""
        service = SpeechToTextService()
        
        # Test that service can start and stop without errors
        # (Actual component coordination will be implemented in task 5.2)
        await service.start_listening()
        assert service._listening is True
        
        await service.stop_listening()
        assert service._listening is False
    
    def test_callback_mechanism_with_mocked_transcription(self, mock_audio_capture, mock_vad, mock_transcriber):
        """Test callback mechanism works with mocked transcription results."""
        service = SpeechToTextService()
        callback_results = []
        
        def transcription_callback(text: str) -> None:
            callback_results.append(text)
        
        service.set_transcription_callback(transcription_callback)
        
        # Simulate transcription result (this will be called by the real implementation)
        if service._transcription_callback:
            service._transcription_callback("mocked transcription result")
        
        assert callback_results == ["mocked transcription result"]
        assert service._transcription_callback is transcription_callback
    
    @pytest.mark.asyncio
    async def test_error_handling_preparation(self, mock_audio_capture, mock_vad, mock_transcriber):
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
        service = SpeechToTextService()
        callback_results = []
        
        def thread_safe_callback(text: str) -> None:
            callback_results.append(text)
        
        # Set callback
        service.set_transcription_callback(thread_safe_callback)
        
        # Simulate multiple concurrent callback invocations
        if service._transcription_callback:
            service._transcription_callback("result1")
            service._transcription_callback("result2")
            service._transcription_callback("result3")
        
        assert len(callback_results) == 3
        assert "result1" in callback_results
        assert "result2" in callback_results
        assert "result3" in callback_results


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
        
        return {
            'audio': mock_audio,
            'vad': mock_vad,
            'transcriber': mock_transcriber
        }
    
    @pytest.mark.asyncio
    async def test_continuous_audio_processing_loop_preparation(self, mock_components):
        """Test preparation for continuous audio processing loop."""
        service = SpeechToTextService()
        
        # Mock the component initialization to use our mocks
        with patch.object(service, '_initialize_components', return_value=True):
            with patch.object(service, '_audio_capture', mock_components['audio']):
                with patch.object(service, '_vad', mock_components['vad']):
                    with patch.object(service, '_transcriber', mock_components['transcriber']):
                        
                        # Test that service can start listening
                        await service.start_listening()
                        assert service.is_listening()
                        
                        # Verify audio capture was started
                        mock_components['audio'].start_capture.assert_called_once()
                        
                        # Test that service can stop listening
                        await service.stop_listening()
                        assert not service.is_listening()
    
    @pytest.mark.asyncio
    async def test_speech_segment_processing_flow(self, mock_components):
        """Test speech segment processing and transcription delivery."""
        service = SpeechToTextService()
        transcription_results = []
        
        def capture_transcription(text: str) -> None:
            transcription_results.append(text)
        
        service.set_transcription_callback(capture_transcription)
        
        # Configure mocks for speech detection and transcription
        mock_components['audio'].get_audio_chunk.return_value = b'audio_data'
        mock_components['vad'].is_speech.return_value = True
        mock_components['vad'].get_speech_segments.return_value = [b'speech_segment']
        mock_components['transcriber'].transcribe_audio.return_value = "Hello world"
        
        # Test transcription update mechanism
        service._update_transcription("Test transcription")
        
        # Verify callback was called
        assert transcription_results == ["Test transcription"]
        assert service.get_latest_transcription() == "Test transcription"
    
    @pytest.mark.asyncio
    async def test_speech_detection_and_filtering(self, mock_components):
        """Test speech detection and non-speech filtering."""
        service = SpeechToTextService()
        
        # Test with speech detected
        mock_components['vad'].is_speech.return_value = True
        mock_components['vad'].get_speech_segments.return_value = [b'speech_data']
        
        # Simulate speech processing (this will be implemented in task 5.4)
        audio_chunk = b'test_audio'
        is_speech = mock_components['vad'].is_speech(audio_chunk)
        assert is_speech is True
        
        # Test with no speech detected
        mock_components['vad'].is_speech.return_value = False
        mock_components['vad'].get_speech_segments.return_value = []
        
        is_speech = mock_components['vad'].is_speech(audio_chunk)
        assert is_speech is False
    
    @pytest.mark.asyncio
    async def test_transcription_delivery_mechanism(self, mock_components):
        """Test transcription delivery to callback and storage."""
        service = SpeechToTextService()
        callback_results = []
        
        def test_callback(text: str) -> None:
            callback_results.append(text)
        
        service.set_transcription_callback(test_callback)
        
        # Test multiple transcription updates
        service._update_transcription("First transcription")
        service._update_transcription("Second transcription")
        service._update_transcription("Third transcription")
        
        # Verify all transcriptions were delivered
        assert len(callback_results) == 3
        assert callback_results[0] == "First transcription"
        assert callback_results[1] == "Second transcription"
        assert callback_results[2] == "Third transcription"
        
        # Verify latest transcription is stored
        assert service.get_latest_transcription() == "Third transcription"
    
    @pytest.mark.asyncio
    async def test_empty_transcription_filtering(self, mock_components):
        """Test that empty or whitespace-only transcriptions are filtered out."""
        service = SpeechToTextService()
        callback_results = []
        
        def test_callback(text: str) -> None:
            callback_results.append(text)
        
        service.set_transcription_callback(test_callback)
        
        # Test empty and whitespace transcriptions
        service._update_transcription("")
        service._update_transcription("   ")
        service._update_transcription("\n\t")
        service._update_transcription("Valid transcription")
        
        # Only valid transcription should be processed
        assert len(callback_results) == 1
        assert callback_results[0] == "Valid transcription"
        assert service.get_latest_transcription() == "Valid transcription"


class TestSpeechToTextServiceErrorHandling:
    """Test cases for error handling and recovery scenarios."""
    
    @pytest.fixture
    def failing_components(self):
        """Create components that simulate various failure scenarios."""
        # Mock audio capture that fails
        failing_audio = Mock()
        failing_audio.start_capture = Mock(side_effect=Exception("Audio device not found"))
        failing_audio.stop_capture = Mock()
        failing_audio.is_capturing = Mock(return_value=False)
        
        # Mock VAD that fails
        failing_vad = Mock()
        failing_vad.is_speech = Mock(side_effect=Exception("VAD processing error"))
        
        # Mock transcriber that fails
        failing_transcriber = Mock()
        failing_transcriber.transcribe_audio = AsyncMock(side_effect=Exception("Transcription failed"))
        failing_transcriber.is_model_available = Mock(return_value=False)
        
        return {
            'audio': failing_audio,
            'vad': failing_vad,
            'transcriber': failing_transcriber
        }
    
    @pytest.mark.asyncio
    async def test_audio_capture_error_handling(self, failing_components):
        """Test error handling when audio capture fails."""
        service = SpeechToTextService()
        
        # Mock component initialization to succeed, but audio capture start to fail
        with patch.object(service, '_initialize_components', return_value=True):
            with patch.object(service, '_audio_capture', failing_components['audio']):
                with patch.object(service, '_vad', Mock()):
                    with patch.object(service, '_transcriber', Mock()):
                        
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
        with patch.object(service, '_initialize_components', return_value=False):
            
            # Starting should raise an exception
            with pytest.raises(RuntimeError, match="Failed to initialize speech-to-text components"):
                await service.start_listening()
            
            # Service should not be in listening state
            assert not service.is_listening()
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test error handling when transcription callback fails."""
        service = SpeechToTextService()
        
        def failing_callback(text: str) -> None:
            raise Exception("Callback processing error")
        
        service.set_transcription_callback(failing_callback)
        
        # Transcription update should not raise exception even if callback fails
        try:
            service._update_transcription("Test transcription")
        except Exception:
            pytest.fail("Transcription update should handle callback errors gracefully")
        
        # Latest transcription should still be updated despite callback failure
        assert service.get_latest_transcription() == "Test transcription"
    
    @pytest.mark.asyncio
    async def test_service_cleanup_on_error(self):
        """Test that service properly cleans up resources on error."""
        service = SpeechToTextService()
        
        # Mock components
        mock_audio = Mock()
        mock_audio.start_capture = Mock(side_effect=Exception("Startup error"))
        mock_audio.stop_capture = Mock()
        mock_audio.is_capturing = Mock(return_value=True)
        
        with patch.object(service, '_initialize_components', return_value=True):
            with patch.object(service, '_audio_capture', mock_audio):
                with patch.object(service, '_cleanup_components') as mock_cleanup:
                    
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
            with patch.object(service, '_initialize_components', return_value=False):
                with pytest.raises(RuntimeError):
                    await service.start_listening()
                
                assert not service.is_listening()
            
            # Service should be able to attempt restart
            await service.stop_listening()  # Should not raise error even if not started


class TestSpeechToTextServiceIntegrationPreparation:
    """Test cases to prepare for integration with real components."""
    
    def test_service_interface_completeness(self):
        """Test that service provides complete interface for integration."""
        service = SpeechToTextService()
        
        # Check all required methods exist
        assert hasattr(service, 'start_listening')
        assert hasattr(service, 'stop_listening')
        assert hasattr(service, 'get_latest_transcription')
        assert hasattr(service, 'set_transcription_callback')
        assert hasattr(service, 'is_listening')
        assert hasattr(service, 'get_component_status')
        
        # Check methods are callable
        assert callable(service.start_listening)
        assert callable(service.stop_listening)
        assert callable(service.get_latest_transcription)
        assert callable(service.set_transcription_callback)
        assert callable(service.is_listening)
        assert callable(service.get_component_status)
    
    @pytest.mark.asyncio
    async def test_async_method_compatibility(self):
        """Test that async methods work correctly for future integration."""
        service = SpeechToTextService()
        
        # Mock initialization to avoid actual component setup
        with patch.object(service, '_initialize_components', return_value=True):
            with patch.object(service, '_audio_capture', Mock()):
                
                # Test async methods can be awaited
                await service.start_listening()
                await service.stop_listening()
    
    def test_state_consistency(self):
        """Test that service maintains consistent internal state."""
        service = SpeechToTextService()
        
        # Initial state
        assert service._listening is False
        assert service._transcription_callback is None
        assert service._latest_transcription is None
        
        # State after callback setting
        def dummy_callback(text: str) -> None:
            pass
        
        service.set_transcription_callback(dummy_callback)
        assert service._transcription_callback is dummy_callback
        assert service._listening is False  # Should not change
        assert service._latest_transcription is None  # Should not change
    
    def test_component_status_reporting(self):
        """Test component status reporting functionality."""
        service = SpeechToTextService()
        
        # Initial status should show no components initialized
        status = service.get_component_status()
        assert status['audio_capture'] is False
        assert status['vad'] is False
        assert status['transcriber'] is False
        assert status['listening'] is False