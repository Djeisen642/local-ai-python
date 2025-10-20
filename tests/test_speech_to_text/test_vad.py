"""Tests for VoiceActivityDetector class."""

from unittest.mock import patch

import pytest

from local_ai.speech_to_text.config import VAD_FRAME_DURATION
from local_ai.speech_to_text.vad import VoiceActivityDetector


@pytest.mark.unit
class TestVoiceActivityDetector:
    """Test cases for VoiceActivityDetector class."""

    def test_vad_initialization_default_parameters(self) -> None:
        """Test VoiceActivityDetector can be initialized with default parameters."""
        vad = VoiceActivityDetector()

        assert vad.sample_rate == 16000
        assert vad.frame_duration == VAD_FRAME_DURATION
        assert hasattr(vad, "vad")  # Should have webrtcvad instance
        assert hasattr(vad, "frame_size")  # Should calculate frame size

    def test_vad_initialization_custom_parameters(self) -> None:
        """Test VoiceActivityDetector can be initialized with custom parameters."""
        vad = VoiceActivityDetector(sample_rate=8000, frame_duration=20)

        assert vad.sample_rate == 8000
        assert vad.frame_duration == 20
        assert hasattr(vad, "vad")
        assert hasattr(vad, "frame_size")

    def test_vad_initialization_invalid_sample_rate(self) -> None:
        """Test VoiceActivityDetector raises error for invalid sample rate."""
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            VoiceActivityDetector(sample_rate=22050)  # Not supported by webrtcvad

    def test_vad_initialization_invalid_frame_duration(self) -> None:
        """Test VoiceActivityDetector raises error for invalid frame duration."""
        with pytest.raises(ValueError, match="Unsupported frame duration"):
            VoiceActivityDetector(frame_duration=25)  # Not supported by webrtcvad

    @pytest.fixture
    def vad(self) -> VoiceActivityDetector:
        """Create a VoiceActivityDetector instance for testing."""
        return VoiceActivityDetector()

    def test_is_speech_with_empty_audio(self, vad: VoiceActivityDetector) -> None:
        """Test is_speech method handles empty audio gracefully."""
        result = vad.is_speech(b"")
        assert result is False, "Should not detect speech in empty audio"

    def test_is_speech_with_invalid_audio_length(
        self, vad: VoiceActivityDetector
    ) -> None:
        """Test is_speech method handles audio with wrong frame size."""
        # Audio that's too short for a proper frame
        short_audio = b"short"
        result = vad.is_speech(short_audio)
        assert result is False, "Should handle short audio gracefully"

    def test_is_speech_with_too_long_audio(self, vad: VoiceActivityDetector) -> None:
        """Test is_speech method handles audio that's too long by truncating."""
        # Create audio that's longer than expected frame size
        expected_bytes = vad.frame_size * 2  # 2 bytes per sample (16-bit)
        long_audio = b"\x00" * (expected_bytes + 100)  # Add extra 100 bytes

        # Should handle long audio by truncating (this tests line 62)
        result = vad.is_speech(long_audio)
        assert isinstance(result, bool), "Should return a boolean result after truncating"

    def test_is_speech_with_vad_exception(self, vad: VoiceActivityDetector) -> None:
        """Test is_speech method handles VAD exceptions gracefully."""
        # Create audio with correct size but invalid format that might cause VAD to fail
        expected_bytes = vad.frame_size * 2
        # Create audio with invalid format (all same byte value might cause issues)
        invalid_audio = b"\xff" * expected_bytes

        # Should handle VAD exceptions gracefully (this tests line 64)
        result = vad.is_speech(invalid_audio)
        assert isinstance(result, bool), (
            "Should return a boolean even if VAD fails internally"
        )

    def test_get_speech_segments_with_empty_buffer(
        self, vad: VoiceActivityDetector
    ) -> None:
        """Test get_speech_segments handles empty buffer gracefully."""
        segments = vad.get_speech_segments([])

        assert segments == [], "Should return empty list for empty buffer"

    def test_get_speech_segments_logic(self, vad: VoiceActivityDetector) -> None:
        """Test get_speech_segments logic by mocking is_speech method."""
        # Create some dummy audio chunks
        chunk1 = b"audio_chunk_1"
        chunk2 = b"audio_chunk_2"
        chunk3 = b"audio_chunk_3"
        chunk4 = b"audio_chunk_4"

        audio_buffer = [chunk1, chunk2, chunk3, chunk4]

        # Mock the is_speech method to return specific results
        with patch.object(vad, "is_speech") as mock_is_speech:
            # Set up mock to return: False, True, True, False
            mock_is_speech.side_effect = [False, True, True, False]

            segments = vad.get_speech_segments(audio_buffer)

            # Should return only chunks 2 and 3 (indices 1 and 2)
            assert segments == [chunk2, chunk3], "Should return only speech segments"

            # Verify is_speech was called for each chunk
            assert mock_is_speech.call_count == 4
            mock_is_speech.assert_any_call(chunk1)
            mock_is_speech.assert_any_call(chunk2)
            mock_is_speech.assert_any_call(chunk3)
            mock_is_speech.assert_any_call(chunk4)

    def test_vad_aggressiveness_configuration(self) -> None:
        """Test VAD uses configured aggressiveness level."""
        vad = VoiceActivityDetector()

        # The VAD should be configured with the aggressiveness from config
        # This will be testable once the implementation is added
        assert hasattr(vad, "vad"), "Should have webrtcvad instance"
        # The actual aggressiveness testing will depend on the implementation

    def test_frame_size_calculation(self) -> None:
        """Test that frame size is calculated correctly for different configurations."""
        # Test standard configuration
        vad1 = VoiceActivityDetector(sample_rate=16000, frame_duration=30)
        expected_frame_size1 = int(16000 * 0.030)  # 30ms at 16kHz
        assert vad1.frame_size == expected_frame_size1

        # Test different configuration
        vad2 = VoiceActivityDetector(sample_rate=8000, frame_duration=20)
        expected_frame_size2 = int(8000 * 0.020)  # 20ms at 8kHz
        assert vad2.frame_size == expected_frame_size2


@pytest.mark.unit
class TestNaturalBreakDetection:
    """Test cases for natural break detection functionality."""

    @pytest.fixture
    def vad(self) -> VoiceActivityDetector:
        """Create a VoiceActivityDetector instance for testing."""
        return VoiceActivityDetector()

    def test_initial_state(self, vad: VoiceActivityDetector) -> None:
        """Test initial state of natural break detection."""
        assert not vad.is_in_speech_segment()
        assert vad.get_current_silence_duration() == 0.0
        assert vad.get_speech_segment_duration() == 0.0
        assert not vad.detect_speech_end()

    def test_reset_silence_timer_starts_speech_segment(
        self, vad: VoiceActivityDetector
    ) -> None:
        """Test that resetting silence timer starts a speech segment."""
        current_time = 1000.0

        # Initially not in speech segment
        assert not vad.is_in_speech_segment()

        # Reset silence timer (simulating speech detection)
        vad.reset_silence_timer(current_time)

        # Should now be in speech segment
        assert vad.is_in_speech_segment()
        assert vad.get_speech_segment_duration(current_time) == 0.0

    def test_start_silence_timer(self, vad: VoiceActivityDetector) -> None:
        """Test starting silence timer."""
        current_time = 1000.0

        # Start speech segment first
        vad.reset_silence_timer(current_time)

        # Start silence timer
        vad.start_silence_timer(current_time + 1.0)

        # Should have silence duration
        assert vad.get_current_silence_duration(current_time + 2.0) == 1.0

    def test_silence_duration_tracking(self, vad: VoiceActivityDetector) -> None:
        """Test silence duration tracking over time."""
        base_time = 1000.0

        # Start speech segment
        vad.reset_silence_timer(base_time)

        # Start silence
        vad.start_silence_timer(base_time + 1.0)

        # Check silence duration at different times
        assert vad.get_current_silence_duration(base_time + 1.5) == 0.5
        assert vad.get_current_silence_duration(base_time + 2.0) == 1.0
        assert vad.get_current_silence_duration(base_time + 3.0) == 2.0

    def test_speech_segment_duration_tracking(self, vad: VoiceActivityDetector) -> None:
        """Test speech segment duration tracking."""
        base_time = 1000.0

        # Start speech segment
        vad.reset_silence_timer(base_time)

        # Check segment duration at different times
        assert vad.get_speech_segment_duration(base_time + 1.0) == 1.0
        assert vad.get_speech_segment_duration(base_time + 5.0) == 5.0
        assert vad.get_speech_segment_duration(base_time + 10.0) == 10.0

    def test_detect_speech_end_short_pause(self, vad: VoiceActivityDetector) -> None:
        """Test that short pauses don't trigger speech end detection."""
        base_time = 1000.0

        # Start speech segment
        vad.reset_silence_timer(base_time)

        # Start silence (short pause)
        vad.start_silence_timer(base_time + 1.0)

        # Check at short pause duration (should not end)
        assert not vad.detect_speech_end(base_time + 1.2)  # 0.2s silence

    def test_detect_speech_end_medium_pause(self, vad: VoiceActivityDetector) -> None:
        """Test that medium pauses trigger speech end detection."""
        base_time = 1000.0

        # Start speech segment
        vad.reset_silence_timer(base_time)

        # Start silence (medium pause)
        vad.start_silence_timer(base_time + 1.0)

        # Check at medium pause duration (should end)
        assert vad.detect_speech_end(base_time + 2.0)  # 1.0s silence (> 0.8s default)

    def test_detect_speech_end_long_pause(self, vad: VoiceActivityDetector) -> None:
        """Test that long pauses definitely trigger speech end detection."""
        base_time = 1000.0

        # Start speech segment
        vad.reset_silence_timer(base_time)

        # Start silence (long pause)
        vad.start_silence_timer(base_time + 1.0)

        # Check at long pause duration (should definitely end)
        assert vad.detect_speech_end(base_time + 4.0)  # 3.0s silence (> 2.0s default)

    def test_detect_speech_end_max_segment_duration(
        self, vad: VoiceActivityDetector
    ) -> None:
        """Test that very long segments are forced to end."""
        base_time = 1000.0

        # Start speech segment
        vad.reset_silence_timer(base_time)

        # No silence, but very long segment
        # Check at max segment duration (should end even without silence)
        assert vad.detect_speech_end(base_time + 31.0)  # 31s > 30s max

    def test_detect_speech_end_not_in_segment(self, vad: VoiceActivityDetector) -> None:
        """Test that speech end detection returns False when not in speech segment."""
        # Not in speech segment
        assert not vad.detect_speech_end()

    def test_finalize_speech_segment(self, vad: VoiceActivityDetector) -> None:
        """Test finalizing speech segment resets state."""
        base_time = 1000.0

        # Start speech segment and silence
        vad.reset_silence_timer(base_time)
        vad.start_silence_timer(base_time + 1.0)

        # Should be in speech segment with silence
        assert vad.is_in_speech_segment()
        assert vad.get_current_silence_duration(base_time + 2.0) > 0

        # Finalize segment
        vad.finalize_speech_segment()

        # Should reset state
        assert not vad.is_in_speech_segment()
        assert vad.get_current_silence_duration() == 0.0
        assert vad.get_speech_segment_duration() == 0.0

    def test_pause_duration_recording(self, vad: VoiceActivityDetector) -> None:
        """Test that pause durations are recorded for adaptation."""
        base_time = 1000.0

        # Simulate several speech-pause cycles
        for i in range(3):
            # Start speech
            vad.reset_silence_timer(base_time + i * 10)

            # Start silence
            vad.start_silence_timer(base_time + i * 10 + 2)

            # Resume speech (this should record the pause duration)
            vad.reset_silence_timer(base_time + i * 10 + 3)  # 1s pause

        # Check that pause durations were recorded (internal state)
        # We can't directly access _recent_pause_durations, but we can test
        # the effect through adaptive thresholds
        vad.get_adaptive_thresholds()

        # Finalize to trigger adaptation
        vad.finalize_speech_segment()

        # Thresholds should exist (may or may not have changed depending on implementation)
        final_thresholds = vad.get_adaptive_thresholds()
        assert "short_pause" in final_thresholds
        assert "medium_pause" in final_thresholds
        assert "long_pause" in final_thresholds

    def test_adaptive_thresholds_initial_values(self, vad: VoiceActivityDetector) -> None:
        """Test that adaptive thresholds start with default values."""
        thresholds = vad.get_adaptive_thresholds()

        # Should start with default values from config
        from local_ai.speech_to_text.config import (
            LONG_PAUSE_THRESHOLD,
            MEDIUM_PAUSE_THRESHOLD,
            SHORT_PAUSE_THRESHOLD,
        )

        assert thresholds["short_pause"] == SHORT_PAUSE_THRESHOLD
        assert thresholds["medium_pause"] == MEDIUM_PAUSE_THRESHOLD
        assert thresholds["long_pause"] == LONG_PAUSE_THRESHOLD

    def test_reset_adaptive_thresholds(self, vad: VoiceActivityDetector) -> None:
        """Test resetting adaptive thresholds to defaults."""
        # Simulate some adaptation by recording pause durations
        base_time = 1000.0

        # Create several short pauses to potentially adapt thresholds
        for i in range(5):
            vad.reset_silence_timer(base_time + i * 5)
            vad.start_silence_timer(base_time + i * 5 + 1)
            vad.reset_silence_timer(base_time + i * 5 + 1.5)  # 0.5s pauses

        vad.finalize_speech_segment()  # Trigger adaptation

        # Reset thresholds
        vad.reset_adaptive_thresholds()

        # Should be back to defaults
        thresholds = vad.get_adaptive_thresholds()
        from local_ai.speech_to_text.config import (
            LONG_PAUSE_THRESHOLD,
            MEDIUM_PAUSE_THRESHOLD,
            SHORT_PAUSE_THRESHOLD,
        )

        assert thresholds["short_pause"] == SHORT_PAUSE_THRESHOLD
        assert thresholds["medium_pause"] == MEDIUM_PAUSE_THRESHOLD
        assert thresholds["long_pause"] == LONG_PAUSE_THRESHOLD

    def test_adaptive_threshold_bounds(self, vad: VoiceActivityDetector) -> None:
        """Test that adaptive thresholds stay within reasonable bounds."""
        base_time = 1000.0

        # Simulate very short pauses (should be clamped)
        for i in range(10):
            vad.reset_silence_timer(base_time + i * 2)
            vad.start_silence_timer(base_time + i * 2 + 0.5)
            vad.reset_silence_timer(base_time + i * 2 + 0.6)  # 0.1s pauses

        vad.finalize_speech_segment()

        thresholds = vad.get_adaptive_thresholds()

        # Thresholds should be within reasonable bounds
        assert 0.1 <= thresholds["short_pause"] <= 1.0
        assert 0.3 <= thresholds["medium_pause"] <= 3.0
        assert 1.0 <= thresholds["long_pause"] <= 5.0

        # Long pause should always be greater than medium pause
        assert thresholds["long_pause"] > thresholds["medium_pause"]

    def test_speech_pattern_scenarios(self, vad: VoiceActivityDetector) -> None:
        """Test different speech pattern scenarios."""
        base_time = 1000.0

        # Scenario 1: Quick command (short speech, medium pause)
        vad.reset_silence_timer(base_time)
        vad.start_silence_timer(base_time + 0.5)  # Short speech

        # Should detect end after medium pause
        assert not vad.detect_speech_end(base_time + 1.0)  # 0.5s silence
        assert vad.detect_speech_end(base_time + 1.5)  # 1.0s silence

        vad.finalize_speech_segment()

        # Scenario 2: Long conversation (long speech, short pauses)
        vad.reset_silence_timer(base_time + 10)

        # Simulate natural pauses during long speech
        vad.start_silence_timer(base_time + 15)
        assert not vad.detect_speech_end(base_time + 15.3)  # Short pause, continue

        vad.reset_silence_timer(base_time + 15.4)  # Resume speech
        vad.start_silence_timer(base_time + 20)
        assert vad.detect_speech_end(base_time + 21)  # Medium pause, end

    def test_default_time_handling(self, vad: VoiceActivityDetector) -> None:
        """Test that methods work with default time (time.time())."""
        # These should not raise exceptions when called without time parameter
        vad.reset_silence_timer()
        vad.start_silence_timer()

        # Should return reasonable values
        assert isinstance(vad.get_current_silence_duration(), float)
        assert isinstance(vad.get_speech_segment_duration(), float)
        assert isinstance(vad.detect_speech_end(), bool)
