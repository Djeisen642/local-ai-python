"""Audio debugging functionality for saving processed audio to WAV files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioDebugger:
    """
    Saves processed audio to WAV files for debugging purposes.

    This class provides a simple way to capture and save audio data that is
    sent to the transcription model. This helps with debugging transcription
    issues and understanding what audio Whisper actually receives.

    Attributes:
        enabled: Whether audio debugging is enabled
        output_dir: Directory where audio files will be saved
    """

    def __init__(
        self,
        enabled: bool = False,
        output_dir: Path | None = None,
    ) -> None:
        """
        Initialize the AudioDebugger.

        Args:
            enabled: Whether to enable audio debugging (default: False)
            output_dir: Directory to save audio files (default: ~/.cache/local_ai/audio_debug)
        """
        self._enabled = enabled

        # Set default output directory if not provided
        if output_dir is None:
            self.output_dir = Path.home() / ".cache" / "local_ai" / "audio_debug"
        else:
            self.output_dir = output_dir

        # Create output directory if debugging is enabled
        if self._enabled:
            self._create_output_directory()

    def is_enabled(self) -> bool:
        """
        Check if audio debugging is enabled.

        Returns:
            True if audio debugging is enabled, False otherwise
        """
        return self._enabled

    def _create_output_directory(self) -> None:
        """
        Create the output directory if it doesn't exist.

        Creates all parent directories as needed. Handles errors gracefully
        by logging them without raising exceptions.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Audio debug directory created: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create audio debug directory {self.output_dir}: {e}")

    def save_audio_sync(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> Path | None:
        """
        Save audio data to a WAV file synchronously.

        Creates a timestamped WAV file with the processed audio data. The filename
        includes the date, time, and duration of the audio.

        Args:
            audio_data: Raw audio data as bytes (16-bit PCM)
            sample_rate: Sample rate in Hz (default: 16000)

        Returns:
            Path to the saved WAV file, or None if debugging is disabled or save failed
        """
        # Do nothing if debugging is disabled
        if not self._enabled:
            return None

        try:
            # Calculate duration from audio data
            # 16-bit audio = 2 bytes per sample
            num_samples = len(audio_data) // 2
            duration_seconds = num_samples / sample_rate
            duration_ms = duration_seconds * 1000

            # Generate timestamped filename with microseconds for uniqueness
            from datetime import datetime

            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y%m%d")
            time_str = timestamp.strftime("%H%M%S")
            microseconds = timestamp.microsecond // 1000  # Convert to milliseconds
            filename = (
                f"audio_{date_str}_{time_str}_{microseconds:03d}_{duration_ms:.1f}ms.wav"
            )

            # Full path to output file
            output_path = self.output_dir / filename

            # Write WAV file
            import wave

            with wave.open(str(output_path), "wb") as wav_file:
                # Set WAV parameters
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(sample_rate)

                # Write audio data
                wav_file.writeframes(audio_data)

            logger.debug(f"Saved audio debug file: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save audio debug file: {e}")
            return None
