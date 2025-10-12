"""Audio capture functionality for speech-to-text."""

import pyaudio
from typing import Optional, List, Dict, Any

from .exceptions import AudioCaptureError, MicrophoneNotFoundError


class AudioCapture:
    """Manages microphone input and audio streaming."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024) -> None:
        """
        Initialize audio capture with specified parameters.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
            
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._capturing = False
        self._pyaudio = None
        self._stream = None
        self._capture_thread = None

    def start_capture(self) -> None:
        """Start capturing audio from microphone."""
        if self._capturing:
            raise AudioCaptureError("Already capturing")
        
        try:
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            
            # Check if default input device is available
            try:
                device_info = self._pyaudio.get_default_input_device_info()
            except OSError as e:
                raise MicrophoneNotFoundError("No microphone found") from e
            
            # Open audio stream
            try:
                self._stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                self._stream.start_stream()
                self._capturing = True
                
            except OSError as e:
                if "Permission denied" in str(e):
                    raise AudioCaptureError("Permission denied") from e
                raise AudioCaptureError(f"Failed to open audio stream: {e}") from e
                
        except Exception as e:
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

    def get_audio_chunk(self) -> Optional[bytes]:
        """
        Get the next audio chunk if available.

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
                        return data
                except OSError as e:
                    raise AudioCaptureError("Failed to read audio") from e
            return None
        except Exception as e:
            raise AudioCaptureError(f"Failed to read audio: {e}") from e



    def is_capturing(self) -> bool:
        """
        Check if currently capturing audio.

        Returns:
            True if capturing, False otherwise
        """
        return self._capturing


