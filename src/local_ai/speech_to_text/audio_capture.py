"""Audio capture functionality for speech-to-text."""

import struct
import time
from typing import Any

import pyaudio

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLE_RATE
from .exceptions import AudioCaptureError, MicrophoneNotFoundError
from .logging_utils import get_logger
from .models import AudioChunk

logger = get_logger(__name__)


class AudioCapture:
    """Manages microphone input and audio streaming."""

    def __init__(
        self,
        sample_rate: int = None,
        chunk_size: int = None,
        enable_filtering: bool = False,
    ) -> None:
        """
        Initialize audio capture with specified parameters.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            enable_filtering: Whether to enable audio filtering pipeline (disabled by
                default - see docs/audio-filtering-evaluation.md)
        """
        self.sample_rate = sample_rate if sample_rate is not None else DEFAULT_SAMPLE_RATE
        self.chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        self._enable_filtering = enable_filtering

        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self._capturing = False
        self._pyaudio = None
        self._stream = None

        # Audio filtering pipeline
        self._audio_filter: Any | None = None

        # Debug tracking
        self._audio_chunks_received = 0
        self._last_audio_level_log = 0
        self._audio_level_log_interval = 5.0  # Log audio levels every 5 seconds

    def start_capture(self) -> None:
        """Start capturing audio from microphone."""
        if self._capturing:
            raise AudioCaptureError("Already capturing")

        try:
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            logger.debug("PyAudio initialized successfully")

            # Log available audio devices for debugging
            self._log_audio_devices()

            # Check if default input device is available
            try:
                device_info = self._pyaudio.get_default_input_device_info()
                try:
                    # Try to log device info, but handle cases where it might be a mock
                    device_name = (
                        device_info.get("name", "Unknown")
                        if hasattr(device_info, "get")
                        else str(device_info)
                    )
                    max_channels = (
                        device_info.get("maxInputChannels", "Unknown")
                        if hasattr(device_info, "get")
                        else "Unknown"
                    )
                    sample_rate = (
                        device_info.get("defaultSampleRate", "Unknown")
                        if hasattr(device_info, "get")
                        else "Unknown"
                    )
                    logger.debug(
                        f"🎤 Default input device found: {device_name} "
                        f"(channels: {max_channels}, sample_rate: {sample_rate})"
                    )
                except (TypeError, AttributeError):
                    logger.debug("🎤 Default input device found (details unavailable)")
            except OSError as e:
                logger.error("❌ No default input device found")
                raise MicrophoneNotFoundError("No microphone found") from e

            # Open audio stream
            try:
                self._stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                )
                self._stream.start_stream()
                self._capturing = True
                logger.debug(
                    f"✅ Audio stream started successfully "
                    f"(sample_rate: {self.sample_rate}, chunk_size: {self.chunk_size})"
                )

                # Reset debug counters
                self._audio_chunks_received = 0
                self._last_audio_level_log = time.time()

            except OSError as e:
                if "Permission denied" in str(e):
                    logger.error("❌ Microphone permission denied")
                    raise AudioCaptureError("Permission denied") from e
                logger.error(f"❌ Failed to open audio stream: {e}")
                raise AudioCaptureError(f"Failed to open audio stream: {e}") from e

        except Exception:
            # Clean up on error
            if self._stream:
                self._stream.close()
                self._stream = None
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
            raise

    def stop_capture(self) -> None:
        """Stop capturing audio."""
        if not self._capturing:
            return

        self._capturing = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

    async def get_audio_chunk(self) -> bytes | None:
        """
        Get the next audio chunk if available, with optional filtering.

        Returns:
            Audio chunk data or None if no data available
        """
        if not self._capturing:
            return None

        try:
            if self._stream and self._stream.is_active():
                # Read from stream in non-blocking mode
                try:
                    data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    if data and len(data) > 0:
                        self._audio_chunks_received += 1

                        # Calculate audio level for debugging
                        audio_level = self._calculate_audio_level(data)

                        # Log audio levels periodically
                        current_time = time.time()
                        if (
                            current_time - self._last_audio_level_log
                            >= self._audio_level_log_interval
                        ):
                            logger.trace(
                                f"🔊 Chunks: {self._audio_chunks_received}, "
                                f"current level: {audio_level:.3f}, "
                                f"chunk size: {len(data)} bytes"
                            )
                            self._last_audio_level_log = current_time

                        # Log significant audio activity
                        if audio_level > 0.01:  # Threshold for "significant" audio
                            logger.trace(
                                f"🎵 Audio activity detected: level={audio_level:.3f}"
                            )

                        # Apply audio filtering if enabled
                        if self._enable_filtering and self._audio_filter:
                            try:
                                # Create AudioChunk for filtering
                                audio_chunk = AudioChunk(
                                    data=data,
                                    timestamp=current_time,
                                    sample_rate=self.sample_rate,
                                    duration=self.chunk_size / self.sample_rate,
                                    is_filtered=False,
                                )

                                # Process through filter pipeline
                                filtered_chunk = (
                                    await self._audio_filter.process_audio_chunk(
                                        audio_chunk
                                    )
                                )
                                return filtered_chunk.data

                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Audio filtering failed, using unfiltered audio: {e}"
                                )
                                # Graceful fallback to unfiltered audio
                                return data

                        return data
                except OSError as e:
                    logger.error(f"❌ Failed to read audio from stream: {e}")
                    raise AudioCaptureError("Failed to read audio") from e
            return None
        except Exception as e:
            logger.error(f"❌ Error in get_audio_chunk: {e}")
            raise AudioCaptureError(f"Failed to read audio: {e}") from e

    def is_capturing(self) -> bool:
        """
        Check if currently capturing audio.

        Returns:
            True if capturing, False otherwise
        """
        return self._capturing

    def _log_audio_devices(self) -> None:
        """Log available audio input devices for debugging."""
        try:
            device_count = self._pyaudio.get_device_count()
            logger.trace(f"🎤 Found {device_count} audio devices:")

            input_devices = []
            for i in range(device_count):
                try:
                    device_info = self._pyaudio.get_device_info_by_index(i)
                    # Handle both real device info and mocks
                    if hasattr(device_info, "get"):
                        max_input_channels = device_info.get("maxInputChannels", 0)
                        device_name = device_info.get("name", f"Device {i}")
                        sample_rate = device_info.get("defaultSampleRate", 0)
                    else:
                        # Fallback for mocks or unexpected formats
                        max_input_channels = getattr(device_info, "maxInputChannels", 0)
                        device_name = getattr(device_info, "name", f"Device {i}")
                        sample_rate = getattr(device_info, "defaultSampleRate", 0)

                    if max_input_channels > 0:
                        input_devices.append(
                            {
                                "index": i,
                                "name": device_name,
                                "channels": max_input_channels,
                                "sample_rate": sample_rate,
                            }
                        )
                        logger.trace(
                            f"  [{i}] {device_name} "
                            f"(in: {max_input_channels}, rate: {sample_rate})"
                        )
                except Exception as e:
                    logger.trace(f"  [{i}] Error getting device info: {e}")

            if input_devices:
                logger.debug(f"✅ Found {len(input_devices)} input devices available")
            else:
                logger.warning("⚠️ No input devices found")

        except Exception as e:
            logger.error(f"❌ Error listing audio devices: {e}")

    def _calculate_audio_level(self, audio_data: bytes) -> float:
        """
        Calculate the audio level (RMS) of the given audio data.

        Args:
            audio_data: Raw audio data bytes

        Returns:
            Audio level as a float between 0.0 and 1.0
        """
        try:
            # Convert bytes to 16-bit integers
            audio_samples = struct.unpack(f"<{len(audio_data) // 2}h", audio_data)

            # Calculate RMS (Root Mean Square)
            if len(audio_samples) > 0:
                rms = (
                    sum(sample**2 for sample in audio_samples) / len(audio_samples)
                ) ** 0.5
                # Normalize to 0-1 range (16-bit audio max value is 32767)
                return min(rms / 32767.0, 1.0)
            return 0.0
        except Exception as e:
            logger.trace(f"Error calculating audio level: {e}")
            return 0.0

    def set_audio_filter(self, audio_filter: Any) -> None:
        """
        Set the audio filtering pipeline.

        Args:
            audio_filter: AudioFilterPipeline instance for processing audio
        """
        self._audio_filter = audio_filter
        logger.debug("Audio filter pipeline set")

    def enable_filtering(self, enabled: bool) -> None:
        """
        Enable or disable audio filtering.

        Args:
            enabled: Whether to enable audio filtering
        """
        self._enable_filtering = enabled
        logger.debug(f"Audio filtering {'enabled' if enabled else 'disabled'}")

    def is_filtering_enabled(self) -> bool:
        """
        Check if audio filtering is enabled.

        Returns:
            True if filtering is enabled, False otherwise
        """
        return self._enable_filtering and self._audio_filter is not None

    def get_debug_stats(self) -> dict[str, Any]:
        """
        Get debug statistics about audio capture.

        Returns:
            Dictionary with debug information
        """
        return {
            "capturing": self._capturing,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "chunks_received": self._audio_chunks_received,
            "stream_active": self._stream.is_active() if self._stream else False,
            "pyaudio_initialized": self._pyaudio is not None,
            "filtering_enabled": self._enable_filtering,
            "filter_available": self._audio_filter is not None,
        }
