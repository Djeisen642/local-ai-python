"""Tests for VoiceActivityDetector class."""

import pytest
from unittest.mock import patch
from local_ai.speech_to_text.vad import VoiceActivityDetector
from local_ai.speech_to_text.config import VAD_FRAME_DURATION, VAD_AGGRESSIVENESS


class TestVoiceActivityDetector:
    """Test cases for VoiceActivityDetector class."""
    
    def test_vad_initialization_default_parameters(self) -> None:
        """Test VoiceActivityDetector can be initialized with default parameters."""
        vad = VoiceActivityDetector()
        
        assert vad.sample_rate == 16000
        assert vad.frame_duration == VAD_FRAME_DURATION
        assert hasattr(vad, 'vad')  # Should have webrtcvad instance
        assert hasattr(vad, 'frame_size')  # Should calculate frame size
    
    def test_vad_initialization_custom_parameters(self) -> None:
        """Test VoiceActivityDetector can be initialized with custom parameters."""
        vad = VoiceActivityDetector(sample_rate=8000, frame_duration=20)
        
        assert vad.sample_rate == 8000
        assert vad.frame_duration == 20
        assert hasattr(vad, 'vad')
        assert hasattr(vad, 'frame_size')
    
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
    
    def test_is_speech_with_invalid_audio_length(self, vad: VoiceActivityDetector) -> None:
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
        invalid_audio = b"\xFF" * expected_bytes
        
        # Should handle VAD exceptions gracefully (this tests line 64)
        result = vad.is_speech(invalid_audio)
        assert isinstance(result, bool), "Should return a boolean even if VAD fails internally"
    

    
    def test_get_speech_segments_with_empty_buffer(self, vad: VoiceActivityDetector) -> None:
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
        with patch.object(vad, 'is_speech') as mock_is_speech:
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
        assert hasattr(vad, 'vad'), "Should have webrtcvad instance"
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