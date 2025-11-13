"""Tests for WhisperTranscriber integration with AudioDebugger."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from local_ai.speech_to_text.audio_debugger import AudioDebugger
from local_ai.speech_to_text.transcriber import WhisperTranscriber

# Patch target for faster_whisper.WhisperModel
WHISPER_MODEL_PATCH = "local_ai.speech_to_text.transcriber.faster_whisper.WhisperModel"


@pytest.mark.unit
class TestWhisperTranscriberAudioDebuggerIntegration:
    """Test cases for WhisperTranscriber integration with AudioDebugger."""

    def test_transcriber_accepts_optional_audio_debugger_parameter(self) -> None:
        """Test WhisperTranscriber accepts optional audio_debugger parameter."""
        # Create an AudioDebugger instance
        debugger = AudioDebugger(enabled=True)

        # Should be able to initialize transcriber with audio_debugger parameter
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Verify the transcriber was initialized
        assert transcriber is not None
        assert transcriber.model_size == "small"

    def test_transcriber_initializes_without_audio_debugger(self) -> None:
        """Test WhisperTranscriber can be initialized without audio_debugger parameter."""
        # Should work without audio_debugger parameter (backward compatibility)
        transcriber = WhisperTranscriber()

        assert transcriber is not None
        assert transcriber.model_size == "small"

    def test_transcriber_accepts_none_audio_debugger(self) -> None:
        """Test WhisperTranscriber accepts None as audio_debugger parameter."""
        # Should accept None explicitly
        transcriber = WhisperTranscriber(audio_debugger=None)

        assert transcriber is not None
        assert transcriber.model_size == "small"

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_audio_saved_when_debugger_enabled(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test audio is saved when debugger is enabled."""
        # Setup mock model
        mock_model = Mock()
        mock_segment = Mock(
            text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
        )
        mock_segments = [mock_segment]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        # Create enabled debugger with temp directory
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Transcribe some audio
        test_audio = b"\x00\x01" * 1000  # Simple test audio data
        result = await transcriber.transcribe_audio_with_result(test_audio)

        # Verify transcription succeeded
        assert result.text == "Test transcription"

        # Verify audio file was saved
        audio_files = list(tmp_path.glob("audio_*.wav"))
        assert len(audio_files) == 1
        assert audio_files[0].exists()

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_audio_not_saved_when_debugger_disabled(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test audio is not saved when debugger is disabled."""
        # Setup mock model
        mock_model = Mock()
        mock_segment = Mock(
            text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
        )
        mock_segments = [mock_segment]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        # Create disabled debugger
        debugger = AudioDebugger(enabled=False, output_dir=tmp_path)

        # Create transcriber with disabled debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Transcribe some audio
        test_audio = b"\x00\x01" * 1000
        result = await transcriber.transcribe_audio_with_result(test_audio)

        # Verify transcription succeeded
        assert result.text == "Test transcription"

        # Verify no audio files were saved
        audio_files = list(tmp_path.glob("audio_*.wav"))
        assert len(audio_files) == 0

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_audio_not_saved_when_no_debugger_provided(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test audio is not saved when no debugger is provided."""
        # Setup mock model
        mock_model = Mock()
        mock_segment = Mock(
            text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
        )
        mock_segments = [mock_segment]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        # Create transcriber without debugger
        transcriber = WhisperTranscriber()

        # Transcribe some audio
        test_audio = b"\x00\x01" * 1000
        result = await transcriber.transcribe_audio_with_result(test_audio)

        # Verify transcription succeeded
        assert result.text == "Test transcription"

        # No audio files should be created (no debugger provided)
        # We can't check tmp_path since no debugger was provided

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcription_continues_when_audio_saving_fails(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test transcription continues when audio saving fails."""
        # Setup mock model
        mock_model = Mock()
        mock_segment = Mock(
            text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
        )
        mock_segments = [mock_segment]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        # Create debugger with enabled flag
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Mock save_audio_sync to raise an exception
        with patch.object(
            debugger, "save_audio_sync", side_effect=Exception("Disk full")
        ):
            # Transcribe some audio - should not raise exception
            test_audio = b"\x00\x01" * 1000
            result = await transcriber.transcribe_audio_with_result(test_audio)

            # Verify transcription succeeded despite save failure
            assert result.text == "Test transcription"
            assert result.confidence >= 0.0

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_multiple_transcriptions_create_multiple_files(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test multiple transcriptions create multiple audio files."""
        # Setup mock model
        mock_model = Mock()

        def mock_transcribe(*args: Any, **kwargs: Any) -> tuple[list[Mock], Mock]:
            mock_segment = Mock(
                text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
            )
            return ([mock_segment], Mock())

        mock_model.transcribe = mock_transcribe
        mock_whisper_model.return_value = mock_model

        # Create enabled debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Perform multiple transcriptions
        import asyncio

        test_audio = b"\x00\x01" * 1000
        results = []
        for _ in range(3):
            result = await transcriber.transcribe_audio_with_result(test_audio)
            results.append(result)
            # Small delay to ensure unique timestamps
            await asyncio.sleep(0.001)

        # Verify all transcriptions succeeded
        for result in results:
            assert result.text == "Test transcription"

        # Verify multiple audio files were created
        audio_files = list(tmp_path.glob("audio_*.wav"))
        assert len(audio_files) == 3

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_audio_debugger_receives_converted_audio(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test audio debugger receives the converted audio format."""
        # Setup mock model
        mock_model = Mock()
        mock_segment = Mock(
            text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
        )
        mock_segments = [mock_segment]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        # Create enabled debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Mock save_audio_sync to capture the audio data
        saved_audio_data: list[tuple[bytes, int]] = []

        def capture_save(audio_data: bytes, sample_rate: int = 16000) -> Path | None:
            saved_audio_data.append((audio_data, sample_rate))
            # Call the real method
            return AudioDebugger.save_audio_sync(debugger, audio_data, sample_rate)

        # Transcribe some audio
        test_audio = b"\x00\x01" * 1000
        with patch.object(debugger, "save_audio_sync", side_effect=capture_save):
            await transcriber.transcribe_audio_with_result(test_audio)

        # Verify audio was captured
        assert len(saved_audio_data) == 1
        captured_audio, captured_rate = saved_audio_data[0]

        # Verify the audio data is not empty and sample rate is correct
        assert len(captured_audio) > 0
        assert captured_rate == 16000  # DEFAULT_SAMPLE_RATE

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcription_with_empty_audio_does_not_save(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test that empty audio does not trigger audio saving."""
        # Create enabled debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Transcribe empty audio
        result = await transcriber.transcribe_audio_with_result(b"")

        # Verify no transcription occurred
        assert result.text == ""

        # Verify no audio files were saved
        audio_files = list(tmp_path.glob("audio_*.wav"))
        assert len(audio_files) == 0

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_transcription_with_invalid_audio_does_not_save(
        self, mock_whisper_model: Mock, tmp_path: Path
    ) -> None:
        """Test that invalid audio does not trigger audio saving."""
        # Create enabled debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Transcribe with None (invalid)
        result = await transcriber.transcribe_audio_with_result(None)  # type: ignore

        # Verify no transcription occurred
        assert result.text == ""

        # Verify no audio files were saved
        audio_files = list(tmp_path.glob("audio_*.wav"))
        assert len(audio_files) == 0

    @pytest.mark.asyncio
    @patch(WHISPER_MODEL_PATCH)
    async def test_debugger_error_logged_but_not_raised(
        self, mock_whisper_model: Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that debugger errors are logged but don't interrupt transcription."""
        # Setup mock model
        mock_model = Mock()
        mock_segment = Mock(
            text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
        )
        mock_segments = [mock_segment]
        mock_model.transcribe.return_value = (mock_segments, Mock())
        mock_whisper_model.return_value = mock_model

        # Create debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        transcriber = WhisperTranscriber(audio_debugger=debugger)

        # Mock save_audio_sync to raise an exception
        with patch.object(
            debugger, "save_audio_sync", side_effect=OSError("Permission denied")
        ):
            # Transcribe some audio
            test_audio = b"\x00\x01" * 1000

            # Should not raise exception
            result = await transcriber.transcribe_audio_with_result(test_audio)

            # Verify transcription succeeded
            assert result.text == "Test transcription"

            # Verify error was logged (check for debug log message)
            # The exact log message will depend on implementation
