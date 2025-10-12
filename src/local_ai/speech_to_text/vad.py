"""Voice Activity Detection for speech-to-text."""

import webrtcvad
from .config import VAD_AGGRESSIVENESS


class VoiceActivityDetector:
    """Detects when speech is present in audio stream."""

    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30) -> None:
        """
        Initialize voice activity detector.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration: Frame duration in milliseconds

        Raises:
            ValueError: If sample_rate or frame_duration is not supported by webrtcvad
        """
        # Validate sample rate (webrtcvad supports 8000, 16000, 32000, 48000 Hz)
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Unsupported sample rate: {sample_rate}. "
                           "WebRTC VAD supports 8000, 16000, 32000, 48000 Hz")
        
        # Validate frame duration (webrtcvad supports 10, 20, 30 ms)
        if frame_duration not in [10, 20, 30]:
            raise ValueError(f"Unsupported frame duration: {frame_duration}. "
                           "WebRTC VAD supports 10, 20, 30 ms")
        
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        
        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad()
        
        # Use optimized aggressiveness if available, otherwise use default
        try:
            from .optimization import get_optimizer
            optimizer = get_optimizer()
            vad_config = optimizer.get_optimized_vad_config()
            aggressiveness = vad_config.get("aggressiveness", VAD_AGGRESSIVENESS)
        except ImportError:
            aggressiveness = VAD_AGGRESSIVENESS
        
        self.vad.set_mode(aggressiveness)

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio data to analyze

        Returns:
            True if speech detected, False otherwise
        """
        # Handle empty or invalid audio
        if not audio_chunk:
            return False
        
        # Check if audio chunk has the correct size for VAD
        expected_bytes = self.frame_size * 2  # 2 bytes per sample (16-bit)
        if len(audio_chunk) != expected_bytes:
            # If audio is too short, pad with zeros or return False
            if len(audio_chunk) < expected_bytes:
                return False
            # If audio is too long, truncate to frame size
            audio_chunk = audio_chunk[:expected_bytes]
        
        try:
            # Use WebRTC VAD to detect speech
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception:
            # If VAD fails for any reason, return False
            return False

    def get_speech_segments(self, audio_buffer: list[bytes]) -> list[bytes]:
        """
        Extract speech segments from audio buffer.

        Args:
            audio_buffer: List of audio chunks to process

        Returns:
            List of speech segments
        """
        if not audio_buffer:
            return []
        
        speech_segments = []
        
        for audio_chunk in audio_buffer:
            if self.is_speech(audio_chunk):
                speech_segments.append(audio_chunk)
        
        return speech_segments
