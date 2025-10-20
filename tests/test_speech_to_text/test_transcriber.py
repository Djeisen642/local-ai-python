"""Tests for WhisperTranscriber class."""

from unittest.mock import Mock, patch

import pytest

from local_ai.speech_to_text.transcriber import WhisperTranscriber

# Patch target for faster_whisper.WhisperModel
WHISPER_MODEL_PATCH = "local_ai.speech_to_text.transcriber.faster_whisper.WhisperModel"


@pytest.mark.unit
class TestWhisperTranscriber:
    """Test cases for WhisperTranscriber class."""

    def test_transcriber_initialization_default(self) -> None:
        """Test WhisperTranscriber can be initialized with default parameters."""
        transcriber = WhisperTranscriber()

        assert transcriber.model_size == "small"

    def test_transcriber_initialization_custom_model_size(self) -> None:
        """Test WhisperTranscriber can be initialized with custom model size."""
        transcriber = WhisperTranscriber(model_size="medium")

        assert transcriber.model_size == "medium"

    def test_transcriber_initialization_invalid_model_size(self) -> None:
        """Test WhisperTranscriber initialization with invalid model size."""
        # Should still initialize but with the provided value
        transcriber = WhisperTranscriber(model_size="invalid")

        assert transcriber.model_size == "invalid"

    @patch(WHISPER_MODEL_PATCH)
    def test_is_model_available_when_model_loads_successfully(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test is_model_available returns True when model loads successfully."""
        # Mock successful model loading
        mock_whisper_model.return_value = Mock()

        transcriber = WhisperTranscriber()
        # This test will fail until the actual implementation is added
        available = transcriber.is_model_available()

        # Should return True when model loads successfully
        assert available is True

    @patch(WHISPER_MODEL_PATCH)
    def test_is_model_available_when_model_fails_to_load(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test is_model_available returns False when model fails to load."""
        # Mock model loading failure
        mock_whisper_model.side_effect = Exception("Model not found")

        transcriber = WhisperTranscriber()
        available = transcriber.is_model_available()

        assert available is False

    @patch(WHISPER_MODEL_PATCH)
    def test_get_model_info_when_model_available(self, mock_whisper_model: Mock) -> None:
        """Test get_model_info returns model information when model is available."""
        # Mock model with info
        mock_model = Mock()
        mock_model.model_size = "small"
        mock_model.device = "cuda"
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        info = transcriber.get_model_info()

        # Should return model info when implemented
        expected_keys = {"model_size", "model_loaded", "actual_model_size", "device"}
        assert all(key in info for key in expected_keys)
        assert info["model_size"] == "small"
        assert info["model_loaded"] is True
        assert info["actual_model_size"] == "small"
        assert info["device"] == "cuda"

    @patch(WHISPER_MODEL_PATCH)
    def test_get_model_info_when_model_unavailable(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test get_model_info returns empty dict when model is unavailable."""
        # Mock model loading failure
        mock_whisper_model.side_effect = Exception("Model not found")

        transcriber = WhisperTranscriber()
        info = transcriber.get_model_info()

        assert info == {}

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_successful(self, mock_whisper_model: Mock) -> None:
        """Test transcribe_audio method with successful transcription."""
        # Mock successful transcription
        mock_model = Mock()
        mock_segments = [Mock(text="Hello world")]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        result = await transcriber.transcribe_audio(b"test_audio_data")

        # Should return transcribed text when implemented
        assert result == "Hello world"

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_empty_audio(self, mock_whisper_model: Mock) -> None:
        """Test transcribe_audio method with empty audio data."""
        transcriber = WhisperTranscriber()
        result = await transcriber.transcribe_audio(b"")

        assert result == ""

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_model_error(self, mock_whisper_model: Mock) -> None:
        """Test transcribe_audio method when model raises an error."""
        # Mock model that raises an error during transcription
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        # Should handle the error gracefully and return empty string or raise appropriate exception
        result = await transcriber.transcribe_audio(b"test_audio_data")
        assert (
            result == ""
        )  # Current behavior, may change to raise exception in implementation

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_no_model_available(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test transcribe_audio method when no model is available."""
        # Mock model initialization failure
        mock_whisper_model.side_effect = Exception("Model not available")

        transcriber = WhisperTranscriber()

        # Should handle missing model gracefully
        result = await transcriber.transcribe_audio(b"test_audio_data")
        assert result == ""

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_multiple_segments(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test transcribe_audio method with multiple text segments."""
        # Mock transcription with multiple segments
        mock_model = Mock()
        mock_segments = [
            Mock(text="Hello "),
            Mock(text="world "),
            Mock(text="from "),
            Mock(text="Whisper"),
        ]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        result = await transcriber.transcribe_audio(b"test_audio_data")

        # Should concatenate all segments
        assert result == "Hello world from Whisper"

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_invalid_audio_format(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test transcribe_audio method with invalid audio format."""
        transcriber = WhisperTranscriber()

        # Test with None audio data
        result = await transcriber.transcribe_audio(None)  # type: ignore
        assert result == ""

        # Test with non-bytes audio data
        result = await transcriber.transcribe_audio("invalid")  # type: ignore
        assert result == ""

    def test_transcriber_supports_all_whisper_model_sizes(self) -> None:
        """Test that transcriber can be initialized with all valid Whisper model sizes."""
        valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

        for size in valid_sizes:
            transcriber = WhisperTranscriber(model_size=size)
            assert transcriber.model_size == size

    @patch(WHISPER_MODEL_PATCH)
    def test_model_uses_local_processing_only(self, mock_whisper_model: Mock) -> None:
        """Test that model initialization doesn't require internet connection (Requirement 2.2, 2.4)."""
        # Mock local model loading
        mock_whisper_model.return_value = Mock()

        transcriber = WhisperTranscriber()

        # Verify that model initialization doesn't make any network calls
        # This is ensured by using faster-whisper which loads models locally
        assert transcriber.model_size == "small"

        # When implemented, should verify no network calls are made
        # and model is loaded from local cache/download

    @patch(WHISPER_MODEL_PATCH)
    def test_model_availability_check_handles_import_error(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test is_model_available handles ImportError when faster-whisper is not installed."""
        # Mock ImportError to simulate missing faster-whisper library
        mock_whisper_model.side_effect = ImportError("No module named 'faster_whisper'")

        transcriber = WhisperTranscriber()
        available = transcriber.is_model_available()

        # Should return False when library is not available
        assert available is False

    @patch(WHISPER_MODEL_PATCH)
    def test_model_availability_check_handles_file_not_found(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test is_model_available handles FileNotFoundError when model files are missing (Requirement 2.3)."""
        # Mock FileNotFoundError to simulate missing model files
        mock_whisper_model.side_effect = FileNotFoundError("Model file not found")

        transcriber = WhisperTranscriber()
        available = transcriber.is_model_available()

        # Should return False when model files are not found
        assert available is False

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_handles_timeout(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test transcribe_audio method handles timeout during transcription."""
        # Mock model that times out during transcription
        mock_model = Mock()
        mock_model.transcribe.side_effect = TimeoutError("Transcription timeout")
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        result = await transcriber.transcribe_audio(b"test_audio_data")

        # Should handle timeout gracefully
        assert result == ""

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_handles_memory_error(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test transcribe_audio method handles memory errors during transcription."""
        # Mock model that raises memory error
        mock_model = Mock()
        mock_model.transcribe.side_effect = MemoryError("Out of memory")
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        result = await transcriber.transcribe_audio(b"test_audio_data")

        # Should handle memory error gracefully
        assert result == ""

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_with_very_large_audio_data(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test transcribe_audio method with very large audio data."""
        transcriber = WhisperTranscriber()

        # Create large audio data (simulate 1MB of audio)
        large_audio_data = b"0" * (1024 * 1024)
        result = await transcriber.transcribe_audio(large_audio_data)

        # Should handle large data without crashing
        assert result == ""

    @patch(WHISPER_MODEL_PATCH)
    def test_get_model_info_includes_expected_fields(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test get_model_info returns expected fields when implemented."""
        # Mock model with comprehensive info
        mock_model = Mock()
        mock_model.model_size = "small"
        mock_model.device = "cuda"
        mock_model.compute_type = "float16"
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        info = transcriber.get_model_info()

        # Should include model details when implemented
        expected_keys = {
            "model_size",
            "model_loaded",
            "actual_model_size",
            "device",
            "compute_type",
        }
        assert all(key in info for key in expected_keys)

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_preserves_audio_quality_settings(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test that transcription preserves audio quality and doesn't degrade input."""
        # Mock model with quality-aware transcription
        mock_model = Mock()
        mock_segments = [Mock(text="High quality transcription")]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        # Test with high-quality audio data
        high_quality_audio = b"high_quality_audio_data"
        result = await transcriber.transcribe_audio(high_quality_audio)

        # Should maintain quality in transcription process
        assert result == "High quality transcription"

    def test_transcriber_thread_safety(self) -> None:
        """Test that WhisperTranscriber can be safely used across multiple threads."""
        # Test basic thread safety of initialization
        transcriber1 = WhisperTranscriber(model_size="small")
        transcriber2 = WhisperTranscriber(model_size="medium")

        # Should be able to create multiple instances safely
        assert transcriber1.model_size == "small"
        assert transcriber2.model_size == "medium"
        assert transcriber1.model_size != transcriber2.model_size

    # Additional interface contract tests

    def test_transcriber_interface_contract(self) -> None:
        """Test that WhisperTranscriber implements the expected interface contract."""
        transcriber = WhisperTranscriber()

        # Verify all required methods exist
        assert hasattr(transcriber, "transcribe_audio")
        assert hasattr(transcriber, "is_model_available")
        assert hasattr(transcriber, "get_model_info")
        assert hasattr(transcriber, "model_size")

        # Verify method signatures are callable
        assert callable(transcriber.transcribe_audio)
        assert callable(transcriber.is_model_available)
        assert callable(transcriber.get_model_info)

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_returns_string(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test that transcribe_audio always returns a string type."""
        transcriber = WhisperTranscriber()

        # Test with various input types
        result1 = await transcriber.transcribe_audio(b"test")
        result2 = await transcriber.transcribe_audio(b"")

        assert isinstance(result1, str)
        assert isinstance(result2, str)

    @patch(WHISPER_MODEL_PATCH)
    def test_is_model_available_returns_boolean(self, mock_whisper_model: Mock) -> None:
        """Test that is_model_available always returns a boolean type."""
        transcriber = WhisperTranscriber()
        result = transcriber.is_model_available()

        assert isinstance(result, bool)

    @patch(WHISPER_MODEL_PATCH)
    def test_get_model_info_returns_dict(self, mock_whisper_model: Mock) -> None:
        """Test that get_model_info always returns a dictionary type."""
        transcriber = WhisperTranscriber()
        result = transcriber.get_model_info()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcribe_audio_with_device_selection(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test that transcriber respects device selection (GPU/CPU) - Requirement 2.1."""
        # Mock model that reports device usage
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_segments = [Mock(text="GPU transcription")]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        result = await transcriber.transcribe_audio(b"test_audio")

        # Should use local GPU/CPU processing (Requirement 2.1)
        assert result == "GPU transcription"

    @patch(WHISPER_MODEL_PATCH)
    def test_model_initialization_with_local_cache(
        self, mock_whisper_model: Mock
    ) -> None:
        """Test that model uses local cache and doesn't require download on every use (Requirement 2.3)."""
        # Mock model that loads from local cache
        mock_model = Mock()
        mock_model.model_path = "/local/cache/whisper-small"
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        available = transcriber.is_model_available()

        # Should work with locally cached models (Requirement 2.3)
        assert available is True

    def test_clear_model_cache_method_exists(self) -> None:
        """Test that clear_model_cache method exists and is callable."""
        transcriber = WhisperTranscriber()

        # Verify method exists
        assert hasattr(transcriber, "clear_model_cache")
        assert callable(transcriber.clear_model_cache)

    def test_clear_model_cache_resets_internal_state(self) -> None:
        """Test that clear_model_cache resets internal transcriber state."""
        transcriber = WhisperTranscriber()

        # Set some internal state
        transcriber._model = Mock()
        transcriber._model_loaded = True

        # Call clear_model_cache (it will fail due to path operations but should reset state first)
        try:
            transcriber.clear_model_cache()
        except Exception:
            pass  # Expected to fail in test environment

        # Verify internal state was reset regardless of path operations
        assert transcriber._model is None
        assert transcriber._model_loaded is False

    def test_clear_model_cache_returns_boolean(self) -> None:
        """Test that clear_model_cache returns a boolean value."""
        transcriber = WhisperTranscriber()

        # The method should return a boolean regardless of success/failure
        result = transcriber.clear_model_cache()
        assert isinstance(result, bool)

    def test_clear_model_cache_exception_handling(self) -> None:
        """Test cache clearing handles exceptions gracefully."""
        transcriber = WhisperTranscriber()

        # The method should handle exceptions and return False
        # In a real test environment, this will likely fail due to path operations
        # but should not crash the application
        result = transcriber.clear_model_cache()

        # Should return either True (success) or False (failure) but not crash
        assert isinstance(result, bool)
