"""Voice Activity Detection for speech-to-text."""

import time
import logging
from .logging_utils import get_logger
from typing import Optional
import webrtcvad

logger = get_logger(__name__)
from .config import (
    VAD_AGGRESSIVENESS, 
    SHORT_PAUSE_THRESHOLD, 
    MEDIUM_PAUSE_THRESHOLD, 
    LONG_PAUSE_THRESHOLD,
    MAX_SEGMENT_DURATION,
    SILENCE_ADAPTATION_FACTOR,
    NOISE_COMPENSATION_THRESHOLD,
    DEFAULT_SAMPLE_RATE,
    VAD_FRAME_DURATION,
    VAD_SUPPORTED_SAMPLE_RATES,
    VAD_SUPPORTED_FRAME_DURATIONS,
    PAUSE_HISTORY_SIZE,
    ADAPTIVE_PAUSE_MIN_SHORT,
    ADAPTIVE_PAUSE_MAX_SHORT,
    ADAPTIVE_PAUSE_MIN_MEDIUM,
    ADAPTIVE_PAUSE_MAX_MEDIUM
)


class VoiceActivityDetector:
    """Detects when speech is present in audio stream."""

    def __init__(self, sample_rate: int = None, frame_duration: int = None) -> None:
        """
        Initialize voice activity detector.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration: Frame duration in milliseconds

        Raises:
            ValueError: If sample_rate or frame_duration is not supported by webrtcvad
        """
        self.sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
        self.frame_duration = frame_duration or VAD_FRAME_DURATION
        
        # Validate sample rate (webrtcvad supports 8000, 16000, 32000, 48000 Hz)
        if self.sample_rate not in VAD_SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}. "
                           f"WebRTC VAD supports {VAD_SUPPORTED_SAMPLE_RATES} Hz")
        
        # Validate frame duration (webrtcvad supports 10, 20, 30 ms)
        if self.frame_duration not in VAD_SUPPORTED_FRAME_DURATIONS:
            raise ValueError(f"Unsupported frame duration: {self.frame_duration}. "
                           f"WebRTC VAD supports {VAD_SUPPORTED_FRAME_DURATIONS} ms")
        
        # Calculate frame size in samples
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
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
        
        # Debug tracking
        self._speech_detections = 0
        self._total_chunks_processed = 0
        self._last_debug_log = time.time()
        self._debug_log_interval = 10.0  # Log VAD stats every 10 seconds
        
        logger.debug(f"🔊 VAD initialized: sample_rate={self.sample_rate}Hz, "
                   f"frame_duration={self.frame_duration}ms, "
                   f"aggressiveness={aggressiveness}, "
                   f"frame_size={self.frame_size} samples")

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio data to analyze

        Returns:
            True if speech detected, False otherwise
        """
        self._total_chunks_processed += 1
        
        # Handle empty or invalid audio
        if not audio_chunk:
            logger.debug("VAD received empty audio chunk")
            return False
        
        # Check if audio chunk has the correct size for VAD
        expected_bytes = self.frame_size * 2  # 2 bytes per sample (16-bit)
        if len(audio_chunk) != expected_bytes:
            # If audio is too short, pad with zeros or return False
            if len(audio_chunk) < expected_bytes:
                logger.debug(f"VAD audio chunk too short: {len(audio_chunk)} < {expected_bytes} bytes")
                return False
            # If audio is too long, truncate to frame size
            logger.debug(f"VAD audio chunk too long: {len(audio_chunk)} > {expected_bytes} bytes, truncating")
            audio_chunk = audio_chunk[:expected_bytes]
        
        try:
            # Use WebRTC VAD to detect speech
            is_speech_detected = self.vad.is_speech(audio_chunk, self.sample_rate)
            
            if is_speech_detected:
                self._speech_detections += 1
                logger.debug(f"🗣️ Speech detected! (detection #{self._speech_detections})")
            
            # Log periodic VAD statistics
            current_time = time.time()
            if current_time - self._last_debug_log >= self._debug_log_interval:
                speech_ratio = (self._speech_detections / self._total_chunks_processed * 100) if self._total_chunks_processed > 0 else 0
                logger.trace(f"🔊 VAD Stats: {self._speech_detections} speech detections in "
                           f"{self._total_chunks_processed} chunks ({speech_ratio:.1f}% speech)")
                self._last_debug_log = current_time
            
            return is_speech_detected
            
        except Exception as e:
            logger.error(f"❌ VAD error processing audio chunk: {e}")
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
        
        # Keep only recent pause durations (last N pauses)
        if len(self._recent_pause_durations) > PAUSE_HISTORY_SIZE:
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
        target_medium = max(ADAPTIVE_PAUSE_MIN_MEDIUM, min(ADAPTIVE_PAUSE_MAX_MEDIUM, avg_pause * 1.2))
        self._adaptive_medium_pause += (target_medium - self._adaptive_medium_pause) * adaptation_rate
        
        # Adjust long pause threshold proportionally
        self._adaptive_long_pause = max(self._adaptive_medium_pause + 0.5, LONG_PAUSE_THRESHOLD)
        
        # Keep short pause threshold relatively stable but slightly adaptive
        target_short = max(ADAPTIVE_PAUSE_MIN_SHORT, min(ADAPTIVE_PAUSE_MAX_SHORT, avg_pause * 0.8))
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
