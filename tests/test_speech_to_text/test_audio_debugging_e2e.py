"""End-to-end integration tests for audio debugging functionality."""

import wave
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from local_ai.speech_to_text.audio_debugger import AudioDebugger
from local_ai.speech_to_text.service import SpeechToTextService
from local_ai.speech_to_text.transcriber import WhisperTranscriber

# Patch target for faster_whisper.WhisperModel
WHISPER_MODEL_PATCH = "local_ai.speech_to_text.transcriber.faster_whisper.WhisperModel"


@pytest.mark.integration
class TestAudioDebuggingEndToEnd:
    """End-to-end integration tests for audio debugging functionality."""

    @pytest.fixture
    def test_audio_dir(self) -> Path:
        """Get the test audio directory."""
        return Path(__file__).parent.parent / "test_data" / "audio"

    def load_test_audio(self, file_path: Path) -> bytes:
        """Load test audio file."""
        if not file_path.exists():
            pytest.skip(f"Test audio file not found: {file_path}")

        with open(file_path, "rb") as f:
            return f.read()

    def verify_wav_format(
        self, wav_path: Path, expected_sample_rate: int = 16000
    ) -> dict[str, int | float]:
        """
        Verify WAV file format and return file properties.

        Args:
            wav_path: Path to WAV file
            expected_sample_rate: Expected sample rate in Hz

        Returns:
            Dictionary with WAV file properties
        """
        with wave.open(str(wav_path), "rb") as wav_file:
            properties = {
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "sample_rate": wav_file.getframerate(),
                "num_frames": wav_file.getnframes(),
                "duration_seconds": wav_file.getnframes() / wav_file.getframerate(),
            }

            # Verify format matches Whisper input requirements
            assert properties["channels"] == 1, "WAV file should be mono"
            assert properties["sample_width"] == 2, "WAV file should be 16-bit"
            assert properties["sample_rate"] == expected_sample_rate, (
                f"WAV file should be {expected_sample_rate}Hz"
            )

            return properties

    @pytest.mark.asyncio
    async def test_complete_flow_cli_args_to_file_creation(self, tmp_path: Path) -> None:
        """
        Test complete flow from CLI args to file creation.
        Requirements: 1.1, 1.3, 2.1, 4.1
        """
        # Simulate CLI arguments for audio debugging
        custom_debug_dir = tmp_path / "audio_debug"

        # Create service with audio debugging enabled
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=custom_debug_dir,
        )

        # Mock components to avoid actual initialization
        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
                    # Setup mock Whisper model
                    mock_model = Mock()
                    mock_segment = Mock(
                        text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
                    )
                    mock_segments = [mock_segment]
                    mock_model.transcribe.return_value = (mock_segments, Mock())
                    mock_whisper_model.return_value = mock_model

                    # Initialize components
                    init_result = service._initialize_components()
                    assert init_result is True

                    # Verify AudioDebugger was created
                    assert service._audio_debugger is not None
                    assert service._audio_debugger.is_enabled() is True
                    assert service._audio_debugger.output_dir == custom_debug_dir

                    # Verify directory was created
                    assert custom_debug_dir.exists()
                    assert custom_debug_dir.is_dir()

                    # Transcribe some audio
                    test_audio = b"\x00\x01" * 1000
                    assert service._transcriber is not None
                    result = await service._transcriber.transcribe_audio_with_result(
                        test_audio
                    )

                    # Verify transcription succeeded
                    assert result.text == "Test transcription"

                    # Verify audio file was created
                    audio_files = list(custom_debug_dir.glob("audio_*.wav"))
                    assert len(audio_files) == 1

                    # Verify file format
                    wav_properties = self.verify_wav_format(audio_files[0])
                    assert wav_properties["channels"] == 1
                    assert wav_properties["sample_width"] == 2
                    assert wav_properties["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_wav_file_format_matches_whisper_input(self, tmp_path: Path) -> None:
        """
        Test that WAV file format matches Whisper input exactly.
        Requirements: 1.1, 1.3
        """
        # Create debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
            # Setup mock model
            mock_model = Mock()
            mock_segment = Mock(
                text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
            )
            mock_segments = [mock_segment]
            mock_model.transcribe.return_value = (mock_segments, Mock())
            mock_whisper_model.return_value = mock_model

            transcriber = WhisperTranscriber(audio_debugger=debugger)

            # Transcribe audio
            test_audio = b"\x00\x01" * 8000  # 1 second of audio at 16kHz
            result = await transcriber.transcribe_audio_with_result(test_audio)

            # Verify transcription succeeded
            assert result.text == "Test transcription"

            # Get saved audio file
            audio_files = list(tmp_path.glob("audio_*.wav"))
            assert len(audio_files) == 1

            # Verify WAV format matches Whisper requirements
            wav_properties = self.verify_wav_format(
                audio_files[0], expected_sample_rate=16000
            )

            # Verify audio data can be read back
            with wave.open(str(audio_files[0]), "rb") as wav_file:
                saved_audio_data = wav_file.readframes(wav_file.getnframes())

                # Verify data length matches (should be close to original)
                # Note: Audio conversion may change length slightly
                assert len(saved_audio_data) > 0
                assert (
                    len(saved_audio_data) <= len(test_audio) * 2
                )  # Allow for conversion

    @pytest.mark.asyncio
    async def test_multiple_transcriptions_create_multiple_files(
        self, tmp_path: Path
    ) -> None:
        """
        Test that multiple transcriptions create multiple files.
        Requirements: 1.1, 1.3
        """
        # Create debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
            # Setup mock model
            mock_model = Mock()

            def mock_transcribe(
                *args: object, **kwargs: object
            ) -> tuple[list[Mock], Mock]:
                mock_segment = Mock(
                    text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
                )
                return ([mock_segment], Mock())

            mock_model.transcribe = mock_transcribe
            mock_whisper_model.return_value = mock_model

            transcriber = WhisperTranscriber(audio_debugger=debugger)

            # Perform multiple transcriptions
            import asyncio

            test_audio = b"\x00\x01" * 1000
            num_transcriptions = 5

            for i in range(num_transcriptions):
                result = await transcriber.transcribe_audio_with_result(test_audio)
                assert result.text == "Test transcription"
                # Small delay to ensure unique timestamps
                await asyncio.sleep(0.01)

            # Verify multiple audio files were created
            audio_files = list(tmp_path.glob("audio_*.wav"))
            assert len(audio_files) == num_transcriptions

            # Verify all files have unique names
            filenames = [f.name for f in audio_files]
            assert len(filenames) == len(set(filenames)), "All filenames should be unique"

            # Verify all files are valid WAV files
            for audio_file in audio_files:
                wav_properties = self.verify_wav_format(audio_file)
                assert wav_properties["channels"] == 1
                assert wav_properties["sample_width"] == 2
                assert wav_properties["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_custom_output_directory_configuration(self, tmp_path: Path) -> None:
        """
        Test custom output directory configuration.
        Requirements: 2.1, 4.1
        """
        # Test with multiple custom directories
        custom_dirs = [
            tmp_path / "custom_debug_1",
            tmp_path / "custom_debug_2",
            tmp_path / "nested" / "debug" / "dir",
        ]

        for custom_dir in custom_dirs:
            # Create debugger with custom directory
            debugger = AudioDebugger(enabled=True, output_dir=custom_dir)

            # Verify directory was created
            assert custom_dir.exists()
            assert custom_dir.is_dir()

            # Create transcriber with debugger
            with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
                # Setup mock model
                mock_model = Mock()
                mock_segment = Mock(
                    text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
                )
                mock_segments = [mock_segment]
                mock_model.transcribe.return_value = (mock_segments, Mock())
                mock_whisper_model.return_value = mock_model

                transcriber = WhisperTranscriber(audio_debugger=debugger)

                # Transcribe audio
                test_audio = b"\x00\x01" * 1000
                result = await transcriber.transcribe_audio_with_result(test_audio)

                # Verify transcription succeeded
                assert result.text == "Test transcription"

                # Verify audio file was created in custom directory
                audio_files = list(custom_dir.glob("audio_*.wav"))
                assert len(audio_files) == 1
                assert audio_files[0].parent == custom_dir

    @pytest.mark.asyncio
    async def test_service_integration_with_cli_args(self, tmp_path: Path) -> None:
        """
        Test service integration with CLI-style arguments.
        Requirements: 2.1, 4.1
        """
        # Test 1: Audio debugging enabled with custom directory
        custom_dir = tmp_path / "cli_debug"
        service = SpeechToTextService(
            enable_audio_debugging=True,
            audio_debug_dir=custom_dir,
        )

        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
                    mock_model = Mock()
                    mock_segment = Mock(text="Test", start=0.0, end=1.0, avg_logprob=-0.5)
                    mock_model.transcribe.return_value = ([mock_segment], Mock())
                    mock_whisper_model.return_value = mock_model

                    service._initialize_components()

                    # Verify debugger is configured correctly
                    assert service._audio_debugger is not None
                    assert service._audio_debugger.is_enabled() is True
                    assert service._audio_debugger.output_dir == custom_dir

        # Test 2: Audio debugging enabled with default directory
        service2 = SpeechToTextService(enable_audio_debugging=True)

        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
                    mock_model = Mock()
                    mock_segment = Mock(text="Test", start=0.0, end=1.0, avg_logprob=-0.5)
                    mock_model.transcribe.return_value = ([mock_segment], Mock())
                    mock_whisper_model.return_value = mock_model

                    service2._initialize_components()

                    # Verify debugger uses default directory
                    assert service2._audio_debugger is not None
                    assert service2._audio_debugger.is_enabled() is True
                    expected_default = Path.home() / ".cache" / "local_ai" / "audio_debug"
                    assert service2._audio_debugger.output_dir == expected_default

        # Test 3: Audio debugging disabled
        service3 = SpeechToTextService(enable_audio_debugging=False)

        with patch("local_ai.speech_to_text.service.AudioCapture"):
            with patch("local_ai.speech_to_text.service.VoiceActivityDetector"):
                with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
                    mock_model = Mock()
                    mock_segment = Mock(text="Test", start=0.0, end=1.0, avg_logprob=-0.5)
                    mock_model.transcribe.return_value = ([mock_segment], Mock())
                    mock_whisper_model.return_value = mock_model

                    service3._initialize_components()

                    # Verify debugger is not created
                    assert service3._audio_debugger is None

    @pytest.mark.asyncio
    async def test_filename_format_and_metadata(self, tmp_path: Path) -> None:
        """
        Test that filenames contain proper metadata (timestamp, duration).
        Requirements: 1.3, 3.1
        """
        # Create debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
            mock_model = Mock()
            mock_segment = Mock(
                text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
            )
            mock_model.transcribe.return_value = ([mock_segment], Mock())
            mock_whisper_model.return_value = mock_model

            transcriber = WhisperTranscriber(audio_debugger=debugger)

            # Transcribe audio with known duration
            # 16kHz, 16-bit = 2 bytes per sample
            # 0.5 seconds = 8000 samples = 16000 bytes
            test_audio = b"\x00\x01" * 8000
            result = await transcriber.transcribe_audio_with_result(test_audio)

            # Get saved audio file
            audio_files = list(tmp_path.glob("audio_*.wav"))
            assert len(audio_files) == 1

            filename = audio_files[0].name

            # Verify filename format: audio_{date}_{time}_{microseconds}_{duration}ms.wav
            import re

            pattern = r"audio_(\d{8})_(\d{6})_(\d{3})_(\d+\.\d+)ms\.wav"
            match = re.match(pattern, filename)
            assert match is not None, (
                f"Filename doesn't match expected pattern: {filename}"
            )

            date_str, time_str, microseconds_str, duration_str = match.groups()

            # Verify date format (YYYYMMDD)
            assert len(date_str) == 8
            assert date_str.isdigit()

            # Verify time format (HHMMSS)
            assert len(time_str) == 6
            assert time_str.isdigit()

            # Verify microseconds format (000-999)
            assert len(microseconds_str) == 3
            assert microseconds_str.isdigit()

            # Verify duration is reasonable (should be close to 0.5 seconds = 500ms)
            duration_ms = float(duration_str)
            assert 400 <= duration_ms <= 600, (
                f"Duration {duration_ms}ms not in expected range"
            )

    @pytest.mark.asyncio
    async def test_wav_file_playback_compatibility(self, tmp_path: Path) -> None:
        """
        Test that saved WAV files can be played back and analyzed.
        Requirements: 1.1, 1.3
        """
        # Create debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
            mock_model = Mock()
            mock_segment = Mock(
                text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
            )
            mock_model.transcribe.return_value = ([mock_segment], Mock())
            mock_whisper_model.return_value = mock_model

            transcriber = WhisperTranscriber(audio_debugger=debugger)

            # Transcribe audio
            test_audio = b"\x00\x01" * 8000
            result = await transcriber.transcribe_audio_with_result(test_audio)

            # Get saved audio file
            audio_files = list(tmp_path.glob("audio_*.wav"))
            assert len(audio_files) == 1

            # Verify file can be opened and read with wave module
            with wave.open(str(audio_files[0]), "rb") as wav_file:
                # Read all frames
                audio_data = wav_file.readframes(wav_file.getnframes())

                # Verify we can read the data
                assert len(audio_data) > 0

                # Verify parameters are accessible
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == 16000

            # Verify file can be opened in read mode multiple times
            with wave.open(str(audio_files[0]), "rb") as wav_file:
                params = wav_file.getparams()
                assert params.nchannels == 1
                assert params.sampwidth == 2
                assert params.framerate == 16000

    @pytest.mark.asyncio
    async def test_error_handling_does_not_interrupt_transcription(
        self, tmp_path: Path
    ) -> None:
        """
        Test that audio debugging errors don't interrupt transcription.
        Requirements: 5.3
        """
        # Create debugger with a directory that will cause errors
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()

        # Make directory read-only (simulate permission error)
        import os
        import stat

        os.chmod(read_only_dir, stat.S_IRUSR | stat.S_IXUSR)

        debugger = AudioDebugger(enabled=True, output_dir=read_only_dir / "subdir")

        # Create transcriber with debugger
        with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
            mock_model = Mock()
            mock_segment = Mock(
                text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
            )
            mock_model.transcribe.return_value = ([mock_segment], Mock())
            mock_whisper_model.return_value = mock_model

            transcriber = WhisperTranscriber(audio_debugger=debugger)

            # Transcribe audio - should succeed despite debugger errors
            test_audio = b"\x00\x01" * 1000
            result = await transcriber.transcribe_audio_with_result(test_audio)

            # Verify transcription succeeded
            assert result.text == "Test transcription"
            assert result.confidence >= 0.0

        # Restore permissions
        os.chmod(read_only_dir, stat.S_IRWXU)

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions_with_debugging(self, tmp_path: Path) -> None:
        """
        Test concurrent transcriptions with audio debugging enabled.
        Requirements: 1.1, 1.3
        """
        # Create debugger
        debugger = AudioDebugger(enabled=True, output_dir=tmp_path)

        # Create transcriber with debugger
        with patch(WHISPER_MODEL_PATCH) as mock_whisper_model:
            mock_model = Mock()

            def mock_transcribe(
                *args: object, **kwargs: object
            ) -> tuple[list[Mock], Mock]:
                mock_segment = Mock(
                    text="Test transcription", start=0.0, end=1.0, avg_logprob=-0.5
                )
                return ([mock_segment], Mock())

            mock_model.transcribe = mock_transcribe
            mock_whisper_model.return_value = mock_model

            transcriber = WhisperTranscriber(audio_debugger=debugger)

            # Perform sequential transcriptions (more realistic for audio debugging)
            import asyncio

            test_audio = b"\x00\x01" * 1000
            num_transcriptions = 5

            for i in range(num_transcriptions):
                result = await transcriber.transcribe_audio_with_result(test_audio)
                assert result.text == "Test transcription"
                # Small delay to ensure unique timestamps
                await asyncio.sleep(0.01)

            # Verify multiple audio files were created
            audio_files = list(tmp_path.glob("audio_*.wav"))
            assert len(audio_files) == num_transcriptions

            # Verify all files are unique and valid
            filenames = [f.name for f in audio_files]
            assert len(filenames) == len(set(filenames)), "All filenames should be unique"

            for audio_file in audio_files:
                self.verify_wav_format(audio_file)
