"""Tests for SpeechToTextService integration with AudioDebugger."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from local_ai.speech_to_text.audio_debugger import AudioDebugger
from local_ai.speech_to_text.service import SpeechToTextService


@pytest.mark.unit
class TestSpeechToTextServiceAudioDebuggerIntegration:
    """Test cases for SpeechToTextService integration with AudioDebugger."""

    def test_service_initialization_with_audio_debugging_enabled(self) -> None:
        """Test service initialization with audio debugging enabled."""
        # This test should fail initially - service doesn't accept debug parameters yet
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=Path("/tmp/test_audio_debug"),
        )

        # Service should store the audio debugging configuration
        assert hasattr(service, "_enable_audio_debugging")
        assert service._enable_audio_debugging is True
        assert hasattr(service, "_audio_debug_dir")
        assert service._audio_debug_dir == Path("/tmp/test_audio_debug")

    def test_service_initialization_with_audio_debugging_disabled(self) -> None:
        """Test service initialization with audio debugging disabled (default)."""
        service = SpeechToTextService()

        # Audio debugging should be disabled by default
        assert hasattr(service, "_enable_audio_debugging")
        assert service._enable_audio_debugging is False

    def test_service_initialization_with_audio_debugging_enabled_default_dir(
        self,
    ) -> None:
        """Test service initialization with audio debugging enabled and default directory."""
        service = SpeechToTextService(enable_audio_debugging=True)

        # Service should use default directory when none specified
        assert service._enable_audio_debugging is True
        assert service._audio_debug_dir is None  # None means use default

    def test_audio_debugger_created_when_enabled(self) -> None:
        """Test that AudioDebugger instance is created when debugging is enabled."""
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=Path("/tmp/test_audio_debug"),
        )

        # Mock other components to avoid actual initialization
        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch("local_ai.speech_to_text.service.WhisperTranscriber"):
                    # Initialize components (this should create AudioDebugger)
                    result = service._initialize_components()

                    assert result is True
                    # After initialization, service should have an AudioDebugger instance
                    assert hasattr(service, "_audio_debugger")
                    assert service._audio_debugger is not None
                    assert isinstance(service._audio_debugger, AudioDebugger)
                    assert service._audio_debugger.is_enabled() is True

    def test_audio_debugger_not_created_when_disabled(self) -> None:
        """Test that AudioDebugger is not created when debugging is disabled."""
        service = SpeechToTextService(enable_audio_debugging=False)

        # Mock component initialization
        with patch.object(service, "_initialize_components") as mock_init:
            mock_init.return_value = True

            # Initialize components
            result = service._initialize_components()

            assert result is True
            # AudioDebugger should not be created when disabled
            assert hasattr(service, "_audio_debugger")
            assert service._audio_debugger is None

    def test_audio_debugger_passed_to_transcriber(self) -> None:
        """Test that AudioDebugger instance is passed to WhisperTranscriber."""
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=Path("/tmp/test_audio_debug"),
        )

        # Mock the WhisperTranscriber constructor to capture arguments
        with patch(
            "local_ai.speech_to_text.service.WhisperTranscriber"
        ) as mock_transcriber_class:
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.is_model_available.return_value = True
            mock_transcriber_class.return_value = mock_transcriber_instance

            # Mock other components
            with patch("local_ai.speech_to_text.service.AudioCapture"):
                with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                    # Initialize components
                    service._initialize_components()

                    # Verify WhisperTranscriber was called with audio_debugger parameter
                    mock_transcriber_class.assert_called_once()
                    call_kwargs = mock_transcriber_class.call_args[1]
                    assert "audio_debugger" in call_kwargs
                    assert isinstance(call_kwargs["audio_debugger"], AudioDebugger)
                    assert call_kwargs["audio_debugger"].is_enabled() is True

    def test_audio_debugger_not_passed_when_disabled(self) -> None:
        """Test that AudioDebugger is not passed to WhisperTranscriber when disabled."""
        service = SpeechToTextService(enable_audio_debugging=False)

        # Mock the WhisperTranscriber constructor to capture arguments
        with patch(
            "local_ai.speech_to_text.service.WhisperTranscriber"
        ) as mock_transcriber_class:
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.is_model_available.return_value = True
            mock_transcriber_class.return_value = mock_transcriber_instance

            # Mock other components
            with patch("local_ai.speech_to_text.service.AudioCapture"):
                with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                    # Initialize components
                    service._initialize_components()

                    # Verify WhisperTranscriber was called with audio_debugger=None
                    mock_transcriber_class.assert_called_once()
                    call_kwargs = mock_transcriber_class.call_args[1]
                    assert "audio_debugger" in call_kwargs
                    assert call_kwargs["audio_debugger"] is None

    def test_audio_debugger_uses_custom_directory(self) -> None:
        """Test that AudioDebugger uses custom directory when specified."""
        custom_dir = Path("/tmp/custom_audio_debug")
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=custom_dir,
        )

        # Mock other components
        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch("local_ai.speech_to_text.service.WhisperTranscriber"):
                    # Initialize components
                    service._initialize_components()

                    # Verify AudioDebugger was created with custom directory
                    assert service._audio_debugger is not None
                    assert service._audio_debugger.output_dir == custom_dir

    def test_audio_debugger_uses_default_directory(self) -> None:
        """Test that AudioDebugger uses default directory when none specified."""
        service = SpeechToTextService(enable_audio_debugging=True)

        # Mock other components
        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch("local_ai.speech_to_text.service.WhisperTranscriber"):
                    # Initialize components
                    service._initialize_components()

                    # Verify AudioDebugger was created with default directory
                    assert service._audio_debugger is not None
                    expected_default = Path.home() / ".cache" / "local_ai" / "audio_debug"
                    assert service._audio_debugger.output_dir == expected_default

    def test_service_initialization_error_handling_with_audio_debugging(self) -> None:
        """Test that service handles errors gracefully when audio debugging is enabled."""
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=Path("/tmp/test_audio_debug"),
        )

        # Mock AudioDebugger to raise an error during creation
        # Need to patch where it's imported (inside the method)
        with patch(
            "local_ai.speech_to_text.audio_debugger.AudioDebugger"
        ) as mock_debugger_class:
            mock_debugger_class.side_effect = Exception("Failed to create debugger")

            # Mock other components
            with patch("local_ai.speech_to_text.service.AudioCapture"):
                with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                    with patch("local_ai.speech_to_text.service.WhisperTranscriber"):
                        # Initialize should handle the error gracefully
                        # (either by continuing without debugging or logging the error)
                        result = service._initialize_components()

                        # Service should still initialize successfully
                        # (audio debugging is optional)
                        assert result is True
                        # AudioDebugger should be None after error
                        assert service._audio_debugger is None

    def test_multiple_services_with_different_debug_settings(self) -> None:
        """Test that multiple service instances can have different debug settings."""
        service1 = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=Path("/tmp/service1_debug"),
        )
        service2 = SpeechToTextService(
            enable_audio_debugging=False,
        )
        service3 = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=Path("/tmp/service3_debug"),
        )

        # Each service should have independent settings
        assert service1._enable_audio_debugging is True
        assert service1._audio_debug_dir == Path("/tmp/service1_debug")

        assert service2._enable_audio_debugging is False

        assert service3._enable_audio_debugging is True
        assert service3._audio_debug_dir == Path("/tmp/service3_debug")
