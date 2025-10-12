"""Voice Activity Detection for speech-to-text."""

import time
from typing import Optional
import webrtcvad
from .config import (
    VAD_AGGRESSIVENESS, 
    SHORT_PAUSE_THRESHOLD, 
    MEDIUM_PAUSE_THRESHOLD, 
    LONG_PAUSE_THRESHOLD,
    MAX_SEGMENT_DURATION,
    SILENCE_ADAPTATION_FACTOR,
    NOISE_COMPENSATION_THRESHOLD
)


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
        
        # Natural break detection state
        self._silence_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._speech_segment_start_time: Optional[float] = None
        self._is_in_speech_segment = False
        
        # Adaptive thresholds (start with defaults, adapt based on speaker patterns)
        self._adaptive_short_pause = SHORT_PAUSE_THRESHOLD
        self._adaptive_medium_pause = MEDIUM_PAUSE_THRESHOLD
        self._adaptive_long_pause = LONG_PAUSE_THRESHOLD
        
        # Speaker pattern tracking
        self._recent_pause_durations: list[float] = []
        self._background_noise_level = 0.0

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

    def detect_speech_end(self, current_time: Optional[float] = None) -> bool:
        """
        Detect if speech has ended based on silence duration and adaptive thresholds.
        
        Args:
            current_time: Current timestamp, uses time.time() if not provided
            
        Returns:
            True if speech segment should be finalized, False otherwise
        """
        if current_time is None:
            current_time = time.time()
            
        # If we're not in a speech segment, no need to detect end
        if not self._is_in_speech_segment:
            return False
            
        # Check for very long segments first (force transcription regardless of silence)
        if (self._speech_segment_start_time is not None and 
            current_time - self._speech_segment_start_time >= MAX_SEGMENT_DURATION):
            return True
            
        # If we don't have a silence start time, we're still hearing speech
        if self._silence_start_time is None:
            return False
            
        silence_duration = current_time - self._silence_start_time
        
        # Check for different types of pauses
        if silence_duration >= self._adaptive_long_pause:
            # Definite end of speech segment
            return True
        elif silence_duration >= self._adaptive_medium_pause:
            # Likely sentence boundary - finalize transcription
            return True
            
        return False

    def reset_silence_timer(self, current_time: Optional[float] = None) -> None:
        """
        Reset the silence timer when speech is detected.
        
        Args:
            current_time: Current timestamp, uses time.time() if not provided
        """
        if current_time is None:
            current_time = time.time()
            
        # If we were in silence, record the pause duration for adaptation
        if self._silence_start_time is not None:
            pause_duration = current_time - self._silence_start_time
            self._record_pause_duration(pause_duration)
            
        self._silence_start_time = None
        self._last_speech_time = current_time
        
        # Start speech segment if not already started
        if not self._is_in_speech_segment:
            self._is_in_speech_segment = True
            self._speech_segment_start_time = current_time

    def start_silence_timer(self, current_time: Optional[float] = None) -> None:
        """
        Start tracking silence duration.
        
        Args:
            current_time: Current timestamp, uses time.time() if not provided
        """
        if current_time is None:
            current_time = time.time()
            
        if self._silence_start_time is None:
            self._silence_start_time = current_time

    def finalize_speech_segment(self) -> None:
        """
        Finalize the current speech segment and reset state.
        """
        self._is_in_speech_segment = False
        self._speech_segment_start_time = None
        self._silence_start_time = None
        
        # Adapt thresholds based on recent patterns
        self._adapt_thresholds()

    def get_current_silence_duration(self, current_time: Optional[float] = None) -> float:
        """
        Get the current silence duration.
        
        Args:
            current_time: Current timestamp, uses time.time() if not provided
            
        Returns:
            Duration of current silence in seconds, 0.0 if not in silence
        """
        if current_time is None:
            current_time = time.time()
            
        if self._silence_start_time is None:
            return 0.0
            
        return current_time - self._silence_start_time

    def is_in_speech_segment(self) -> bool:
        """
        Check if currently in an active speech segment.
        
        Returns:
            True if in speech segment, False otherwise
        """
        return self._is_in_speech_segment

    def get_speech_segment_duration(self, current_time: Optional[float] = None) -> float:
        """
        Get the duration of the current speech segment.
        
        Args:
            current_time: Current timestamp, uses time.time() if not provided
            
        Returns:
            Duration of current speech segment in seconds, 0.0 if not in segment
        """
        if current_time is None:
            current_time = time.time()
            
        if not self._is_in_speech_segment or self._speech_segment_start_time is None:
            return 0.0
            
        return current_time - self._speech_segment_start_time

    def _record_pause_duration(self, duration: float) -> None:
        """
        Record a pause duration for adaptive threshold adjustment.
        
        Args:
            duration: Pause duration in seconds
        """
        self._recent_pause_durations.append(duration)
        
        # Keep only recent pause durations (last 20 pauses)
        if len(self._recent_pause_durations) > 20:
            self._recent_pause_durations.pop(0)

    def _adapt_thresholds(self) -> None:
        """
        Adapt pause thresholds based on speaker patterns.
        """
        if len(self._recent_pause_durations) < 3:
            return  # Need at least 3 samples to adapt
            
        # Calculate average pause duration
        avg_pause = sum(self._recent_pause_durations) / len(self._recent_pause_durations)
        
        # Adapt thresholds gradually
        adaptation_rate = SILENCE_ADAPTATION_FACTOR
        
        # Adjust medium pause threshold based on speaker's natural rhythm
        target_medium = max(0.5, min(1.5, avg_pause * 1.2))  # Clamp between 0.5-1.5s
        self._adaptive_medium_pause += (target_medium - self._adaptive_medium_pause) * adaptation_rate
        
        # Adjust long pause threshold proportionally
        self._adaptive_long_pause = max(self._adaptive_medium_pause + 0.5, LONG_PAUSE_THRESHOLD)
        
        # Keep short pause threshold relatively stable but slightly adaptive
        target_short = max(0.2, min(0.5, avg_pause * 0.8))  # Clamp between 0.2-0.5s
        self._adaptive_short_pause += (target_short - self._adaptive_short_pause) * adaptation_rate * 0.5

    def get_adaptive_thresholds(self) -> dict[str, float]:
        """
        Get current adaptive threshold values.
        
        Returns:
            Dictionary with current threshold values
        """
        return {
            "short_pause": self._adaptive_short_pause,
            "medium_pause": self._adaptive_medium_pause,
            "long_pause": self._adaptive_long_pause
        }

    def reset_adaptive_thresholds(self) -> None:
        """
        Reset adaptive thresholds to default values.
        """
        self._adaptive_short_pause = SHORT_PAUSE_THRESHOLD
        self._adaptive_medium_pause = MEDIUM_PAUSE_THRESHOLD
        self._adaptive_long_pause = LONG_PAUSE_THRESHOLD
        self._recent_pause_durations.clear()
