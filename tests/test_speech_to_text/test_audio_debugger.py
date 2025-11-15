"""Tests for AudioDebugger class."""

import tempfile
from pathlib import Path

import pytest

from local_ai.speech_to_text.audio_debugger import AudioDebugger


@pytest.mark.unit
class TestAudioDebuggerInitialization:
    """Test cases for AudioDebugger initialization and basic methods."""

    def test_initialization_disabled_by_default(self) -> None:
        """Test AudioDebugger is disabled by default."""
        debugger = AudioDebugger()

        assert not debugger.is_enabled()

    def test_initialization_enabled_explicitly(self) -> None:
        """Test AudioDebugger can be enabled explicitly."""
        debugger = AudioDebugger(enabled=True)

        assert debugger.is_enabled()

    def test_initialization_disabled_explicitly(self) -> None:
        """Test AudioDebugger can be disabled explicitly."""
        debugger = AudioDebugger(enabled=False)

        assert not debugger.is_enabled()

    def test_initialization_with_default_output_dir(self) -> None:
        """Test AudioDebugger uses default output directory when not specified."""
        debugger = AudioDebugger(enabled=True)

        # Should have an output_dir attribute
        assert hasattr(debugger, "output_dir")
        assert isinstance(debugger.output_dir, Path)

    def test_initialization_with_custom_output_dir(self) -> None:
        """Test AudioDebugger accepts custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=custom_dir)

            assert debugger.output_dir == custom_dir

    def test_directory_creation_on_initialization_when_enabled(self) -> None:
        """Test AudioDebugger creates output directory on initialization when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug_test"

            # Directory should not exist yet
            assert not output_dir.exists()

            # Create debugger with enabled=True
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Directory should now exist
            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_directory_not_created_when_disabled(self) -> None:
        """Test AudioDebugger does not create directory when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug_test"

            # Directory should not exist yet
            assert not output_dir.exists()

            # Create debugger with enabled=False
            debugger = AudioDebugger(enabled=False, output_dir=output_dir)

            # Directory should still not exist
            assert not output_dir.exists()

    def test_directory_creation_handles_existing_directory(self) -> None:
        """Test AudioDebugger handles existing output directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "existing_dir"
            output_dir.mkdir()

            # Should not raise exception when directory already exists
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_directory_creation_handles_nested_paths(self) -> None:
        """Test AudioDebugger creates nested directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "level1" / "level2" / "audio_debug"

            # Nested path should not exist yet
            assert not output_dir.exists()

            # Create debugger
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # All nested directories should be created
            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_is_enabled_method_returns_boolean(self) -> None:
        """Test is_enabled method returns boolean value."""
        debugger_enabled = AudioDebugger(enabled=True)
        debugger_disabled = AudioDebugger(enabled=False)

        assert isinstance(debugger_enabled.is_enabled(), bool)
        assert isinstance(debugger_disabled.is_enabled(), bool)


@pytest.mark.unit
class TestAudioDebuggerSaving:
    """Test cases for AudioDebugger audio file saving functionality."""

    def test_save_audio_sync_creates_wav_file(self) -> None:
        """Test save_audio_sync creates a WAV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create some dummy audio data (1 second of silence at 16kHz, 16-bit)
            sample_rate = 16000
            duration_seconds = 1.0
            num_samples = int(sample_rate * duration_seconds)
            # 16-bit audio = 2 bytes per sample
            audio_data = b"\x00\x00" * num_samples

            # Save audio
            result_path = debugger.save_audio_sync(audio_data, sample_rate=sample_rate)

            # Should return a Path object
            assert result_path is not None
            assert isinstance(result_path, Path)

            # File should exist
            assert result_path.exists()
            assert result_path.is_file()

            # File should have .wav extension
            assert result_path.suffix == ".wav"

    def test_save_audio_sync_returns_none_when_disabled(self) -> None:
        """Test save_audio_sync returns None when debugging is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=False, output_dir=output_dir)

            # Create some dummy audio data
            audio_data = b"\x00\x00" * 16000

            # Save audio (should do nothing)
            result_path = debugger.save_audio_sync(audio_data, sample_rate=16000)

            # Should return None
            assert result_path is None

            # No files should be created
            if output_dir.exists():
                assert len(list(output_dir.iterdir())) == 0

    def test_save_audio_sync_filename_format(self) -> None:
        """Test save_audio_sync generates correctly formatted filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create audio data (0.5 seconds at 16kHz)
            sample_rate = 16000
            duration_seconds = 0.5
            num_samples = int(sample_rate * duration_seconds)
            audio_data = b"\x00\x00" * num_samples

            # Save audio
            result_path = debugger.save_audio_sync(audio_data, sample_rate=sample_rate)

            # Filename should match pattern: audio_{YYYYMMDD}_{HHMMSS}_{duration_ms}.wav
            assert result_path is not None
            filename = result_path.name

            # Should start with "audio_"
            assert filename.startswith("audio_")

            # Should end with ".wav"
            assert filename.endswith(".wav")

            # Should contain duration in milliseconds (500ms for 0.5 seconds)
            assert "500ms" in filename or "500.0ms" in filename

            # Should contain date and time components (8 digits for date, 6 for time)
            parts = filename.replace(".wav", "").split("_")
            assert len(parts) >= 4  # audio, date, time, duration

    def test_save_audio_sync_duration_calculation(self) -> None:
        """Test save_audio_sync correctly calculates duration from audio data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Test different durations
            test_cases = [
                (16000, 0.5, "500"),  # 0.5 seconds = 500ms
                (16000, 1.0, "1000"),  # 1.0 seconds = 1000ms
                (16000, 2.5, "2500"),  # 2.5 seconds = 2500ms
            ]

            for sample_rate, duration_seconds, expected_ms in test_cases:
                num_samples = int(sample_rate * duration_seconds)
                audio_data = b"\x00\x00" * num_samples

                result_path = debugger.save_audio_sync(
                    audio_data, sample_rate=sample_rate
                )
                assert result_path is not None
                filename = result_path.name

                # Check that duration is in filename
                assert expected_ms in filename, (
                    f"Expected {expected_ms}ms in filename {filename}"
                )

    def test_save_audio_sync_wav_format_validation(self) -> None:
        """Test save_audio_sync creates valid WAV files with correct format."""
        import wave

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create audio data
            sample_rate = 16000
            num_samples = 16000  # 1 second
            audio_data = b"\x00\x00" * num_samples

            # Save audio
            result_path = debugger.save_audio_sync(audio_data, sample_rate=sample_rate)

            # Open and validate WAV file
            with wave.open(str(result_path), "rb") as wav_file:
                # Check sample rate
                assert wav_file.getframerate() == sample_rate

                # Check channels (should be mono)
                assert wav_file.getnchannels() == 1

                # Check sample width (should be 16-bit = 2 bytes)
                assert wav_file.getsampwidth() == 2

                # Check number of frames
                assert wav_file.getnframes() == num_samples

    def test_save_audio_sync_multiple_files(self) -> None:
        """Test save_audio_sync creates multiple unique files."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create and save multiple audio files
            paths = []
            for i in range(3):
                audio_data = b"\x00\x00" * 16000
                result_path = debugger.save_audio_sync(audio_data, sample_rate=16000)
                paths.append(result_path)

                # Small delay to ensure different timestamps
                time.sleep(0.01)

            # All paths should be unique
            assert len(paths) == len(set(paths))

            # All files should exist
            for path in paths:
                assert path is not None
                assert path.exists()

    def test_save_audio_sync_with_different_sample_rates(self) -> None:
        """Test save_audio_sync handles different sample rates correctly."""
        import wave

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Test different sample rates
            sample_rates = [8000, 16000, 32000, 48000]

            for sample_rate in sample_rates:
                num_samples = sample_rate  # 1 second
                audio_data = b"\x00\x00" * num_samples

                result_path = debugger.save_audio_sync(
                    audio_data, sample_rate=sample_rate
                )

                # Validate WAV file has correct sample rate
                with wave.open(str(result_path), "rb") as wav_file:
                    assert wav_file.getframerate() == sample_rate


@pytest.mark.unit
class TestAudioDebuggerLogging:
    """Test cases for AudioDebugger logging functionality."""

    def test_logs_sample_rate_when_enabled(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sample rate and duration are logged when AUDIO_DEBUG_LOG_SAMPLE_RATES is True."""
        import logging

        # Enable sample rate logging
        from local_ai.speech_to_text import config

        monkeypatch.setattr(config, "AUDIO_DEBUG_LOG_SAMPLE_RATES", True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create audio data (1 second at 16kHz)
            sample_rate = 16000
            duration_seconds = 1.0
            num_samples = int(sample_rate * duration_seconds)
            audio_data = b"\x00\x00" * num_samples

            with caplog.at_level(logging.INFO):
                result_path = debugger.save_audio_sync(
                    audio_data, sample_rate=sample_rate
                )

            # Should have logged sample rate and duration
            assert result_path is not None
            assert any(
                "sample_rate=16000Hz" in record.message
                and "duration=1.000s" in record.message
                for record in caplog.records
            )

    def test_does_not_log_sample_rate_when_disabled(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sample rate is not logged when AUDIO_DEBUG_LOG_SAMPLE_RATES is False."""
        import logging

        # Disable sample rate logging (default)
        from local_ai.speech_to_text import config

        monkeypatch.setattr(config, "AUDIO_DEBUG_LOG_SAMPLE_RATES", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create audio data
            sample_rate = 16000
            audio_data = b"\x00\x00" * 16000

            with caplog.at_level(logging.INFO):
                result_path = debugger.save_audio_sync(
                    audio_data, sample_rate=sample_rate
                )

            # Should not have logged sample rate info
            assert result_path is not None
            assert not any(
                "sample_rate=" in record.message and "duration=" in record.message
                for record in caplog.records
                if record.levelname == "INFO"
            )

    def test_logs_different_sample_rates(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that different sample rates are logged correctly."""
        import logging

        # Enable sample rate logging
        from local_ai.speech_to_text import config

        monkeypatch.setattr(config, "AUDIO_DEBUG_LOG_SAMPLE_RATES", True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Test different sample rates
            test_cases = [
                (8000, 0.5, "sample_rate=8000Hz", "duration=0.500s"),
                (16000, 1.0, "sample_rate=16000Hz", "duration=1.000s"),
                (48000, 2.0, "sample_rate=48000Hz", "duration=2.000s"),
            ]

            for (
                sample_rate,
                duration_seconds,
                expected_rate,
                expected_duration,
            ) in test_cases:
                caplog.clear()
                num_samples = int(sample_rate * duration_seconds)
                audio_data = b"\x00\x00" * num_samples

                with caplog.at_level(logging.INFO):
                    result_path = debugger.save_audio_sync(
                        audio_data, sample_rate=sample_rate
                    )

                # Should have logged the correct sample rate and duration
                assert result_path is not None
                assert any(
                    expected_rate in record.message
                    and expected_duration in record.message
                    for record in caplog.records
                )


@pytest.mark.unit
class TestAudioDebuggerErrorHandling:
    """Test cases for AudioDebugger error handling."""

    def test_handles_non_writable_directory(self) -> None:
        """Test AudioDebugger handles non-writable directories gracefully."""
        import os
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "readonly_dir"
            output_dir.mkdir()

            # Make directory read-only
            os.chmod(output_dir, stat.S_IRUSR | stat.S_IXUSR)

            try:
                debugger = AudioDebugger(enabled=True, output_dir=output_dir)

                # Create audio data
                audio_data = b"\x00\x00" * 16000

                # Should not raise exception, just return None
                result = debugger.save_audio_sync(audio_data, sample_rate=16000)

                # Should return None on error
                assert result is None

            finally:
                # Restore write permissions for cleanup
                os.chmod(output_dir, stat.S_IRWXU)

    def test_handles_invalid_audio_data(self) -> None:
        """Test AudioDebugger handles invalid audio data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Test with empty audio data
            result = debugger.save_audio_sync(b"", sample_rate=16000)

            # Should handle gracefully (either return None or create empty file)
            # The important thing is it doesn't raise an exception
            assert result is None or isinstance(result, Path)

    def test_handles_odd_length_audio_data(self) -> None:
        """Test AudioDebugger handles odd-length audio data (not divisible by 2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # Create audio data with odd number of bytes (invalid for 16-bit audio)
            audio_data = b"\x00" * 16001  # Odd number

            # Should handle gracefully without raising exception
            result = debugger.save_audio_sync(audio_data, sample_rate=16000)

            # Should either succeed or return None, but not crash
            assert result is None or isinstance(result, Path)

    def test_error_logging_without_exceptions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that errors are logged but don't raise exceptions."""
        import logging
        import os
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "readonly_dir"
            output_dir.mkdir()

            # Make directory read-only
            os.chmod(output_dir, stat.S_IRUSR | stat.S_IXUSR)

            try:
                with caplog.at_level(logging.ERROR):
                    debugger = AudioDebugger(enabled=True, output_dir=output_dir)
                    audio_data = b"\x00\x00" * 16000

                    # Should not raise exception
                    result = debugger.save_audio_sync(audio_data, sample_rate=16000)

                    # Should return None
                    assert result is None

                    # Should have logged an error
                    assert any(
                        "Failed to save audio debug file" in record.message
                        for record in caplog.records
                    )

            finally:
                # Restore write permissions for cleanup
                os.chmod(output_dir, stat.S_IRWXU)

    def test_handles_directory_creation_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test AudioDebugger handles directory creation failures gracefully."""
        import logging
        import os
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file where we want to create a directory
            parent_dir = Path(tmpdir) / "parent"
            parent_dir.mkdir()

            # Create a file with the name we want for our directory
            blocking_file = parent_dir / "audio_debug"
            blocking_file.write_text("blocking")

            # Make parent directory read-only to prevent deletion
            os.chmod(parent_dir, stat.S_IRUSR | stat.S_IXUSR)

            try:
                with caplog.at_level(logging.ERROR):
                    # This should fail to create the directory but not raise exception
                    debugger = AudioDebugger(enabled=True, output_dir=blocking_file)

                    # Should have logged an error about directory creation
                    assert any(
                        "Failed to create audio debug directory" in record.message
                        for record in caplog.records
                    )

            finally:
                # Restore write permissions for cleanup
                os.chmod(parent_dir, stat.S_IRWXU)

    def test_continues_after_save_failure(self) -> None:
        """Test that AudioDebugger can continue after a save failure."""
        import os
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "audio_debug"
            debugger = AudioDebugger(enabled=True, output_dir=output_dir)

            # First save should succeed
            audio_data = b"\x00\x00" * 16000
            result1 = debugger.save_audio_sync(audio_data, sample_rate=16000)
            assert result1 is not None
            assert result1.exists()

            # Make directory read-only (no write or execute permissions)
            os.chmod(output_dir, stat.S_IRUSR)

            try:
                # Second save should fail gracefully
                result2 = debugger.save_audio_sync(audio_data, sample_rate=16000)
                assert result2 is None

            finally:
                # Restore write permissions
                os.chmod(output_dir, stat.S_IRWXU)

            # Third save should succeed again
            result3 = debugger.save_audio_sync(audio_data, sample_rate=16000)
            assert result3 is not None
            assert result3.exists()
