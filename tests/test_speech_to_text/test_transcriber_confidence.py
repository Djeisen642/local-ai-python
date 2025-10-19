"""Tests for WhisperTranscriber confidence rating functionality."""

import pytest
from unittest.mock import Mock, MagicMock
from local_ai.speech_to_text.transcriber import WhisperTranscriber
from local_ai.speech_to_text.models import TranscriptionResult
from local_ai.speech_to_text.config import (
    CONFIDENCE_LOGPROB_MIN,
    CONFIDENCE_LOGPROB_MAX
)


@pytest.mark.unit
class TestWhisperTranscriberConfidence:
    """Test cases for confidence rating functionality in WhisperTranscriber."""

    def test_calculate_confidence_empty_segments(self) -> None:
        """Test confidence calculation with empty segments list."""
        transcriber = WhisperTranscriber()
        confidence = transcriber._calculate_confidence([])
        
        assert confidence == 0.0

    def test_calculate_confidence_single_segment_high_confidence(self) -> None:
        """Test confidence calculation with single high-confidence segment."""
        transcriber = WhisperTranscriber()
        
        # Create mock segment with high confidence (avg_logprob close to max)
        mock_segment = Mock()
        mock_segment.avg_logprob = -0.2  # High confidence
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        
        confidence = transcriber._calculate_confidence([mock_segment])
        
        # Should be close to 1.0 for high confidence
        assert confidence > 0.9
        assert confidence <= 1.0

    def test_calculate_confidence_single_segment_low_confidence(self) -> None:
        """Test confidence calculation with single low-confidence segment."""
        transcriber = WhisperTranscriber()
        
        # Create mock segment with low confidence (avg_logprob close to min)
        mock_segment = Mock()
        mock_segment.avg_logprob = -1.8  # Low confidence
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        
        confidence = transcriber._calculate_confidence([mock_segment])
        
        # Should be close to 0.0 for low confidence
        assert confidence < 0.2
        assert confidence >= 0.0

    def test_calculate_confidence_single_segment_medium_confidence(self) -> None:
        """Test confidence calculation with single medium-confidence segment."""
        transcriber = WhisperTranscriber()
        
        # Create mock segment with medium confidence
        mock_segment = Mock()
        mock_segment.avg_logprob = -1.0  # Medium confidence
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        
        confidence = transcriber._calculate_confidence([mock_segment])
        
        # Should be in middle range
        assert 0.4 < confidence < 0.7

    def test_calculate_confidence_multiple_segments_weighted_average(self) -> None:
        """Test confidence calculation with multiple segments uses weighted average."""
        transcriber = WhisperTranscriber()
        
        # Create segments with different durations and confidence levels
        segment1 = Mock()
        segment1.avg_logprob = -0.3  # High confidence
        segment1.start = 0.0
        segment1.end = 1.0  # 1 second duration
        
        segment2 = Mock()
        segment2.avg_logprob = -1.5  # Low confidence  
        segment2.start = 1.0
        segment2.end = 4.0  # 3 seconds duration (longer)
        
        confidence = transcriber._calculate_confidence([segment1, segment2])
        
        # Should be weighted toward the longer, lower-confidence segment
        assert 0.0 <= confidence <= 1.0
        # The longer segment has lower confidence, so overall should be lower
        assert confidence < 0.6

    def test_calculate_confidence_segments_without_timing_info(self) -> None:
        """Test confidence calculation when segments lack timing information."""
        transcriber = WhisperTranscriber()
        
        # Create segment without start/end attributes
        mock_segment = Mock()
        mock_segment.avg_logprob = -0.5
        # Remove timing attributes to simulate missing data
        if hasattr(mock_segment, 'start'):
            delattr(mock_segment, 'start')
        if hasattr(mock_segment, 'end'):
            delattr(mock_segment, 'end')
        
        confidence = transcriber._calculate_confidence([mock_segment])
        
        # Should handle missing timing gracefully and return 0.0
        assert confidence == 0.0

    def test_calculate_confidence_segments_without_logprob(self) -> None:
        """Test confidence calculation when segments lack avg_logprob."""
        transcriber = WhisperTranscriber()
        
        # Create segment without avg_logprob attribute
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        # Remove avg_logprob to simulate missing data
        if hasattr(mock_segment, 'avg_logprob'):
            delattr(mock_segment, 'avg_logprob')
        
        confidence = transcriber._calculate_confidence([mock_segment])
        
        # Should handle missing logprob gracefully and return 0.0
        assert confidence == 0.0

    def test_calculate_confidence_zero_duration_segments(self) -> None:
        """Test confidence calculation with zero-duration segments."""
        transcriber = WhisperTranscriber()
        
        # Create segment with zero duration
        mock_segment = Mock()
        mock_segment.avg_logprob = -0.5
        mock_segment.start = 1.0
        mock_segment.end = 1.0  # Same start and end time
        
        confidence = transcriber._calculate_confidence([mock_segment])
        
        # Should handle zero duration gracefully
        assert confidence == 0.0

    def test_calculate_confidence_extreme_logprob_values(self) -> None:
        """Test confidence calculation with extreme avg_logprob values."""
        transcriber = WhisperTranscriber()
        
        # Test with value below minimum expected range
        segment_below_min = Mock()
        segment_below_min.avg_logprob = -5.0  # Below CONFIDENCE_LOGPROB_MIN
        segment_below_min.start = 0.0
        segment_below_min.end = 1.0
        
        confidence_below = transcriber._calculate_confidence([segment_below_min])
        assert confidence_below == 0.0  # Should clamp to 0.0
        
        # Test with value above maximum expected range
        segment_above_max = Mock()
        segment_above_max.avg_logprob = 0.5  # Above CONFIDENCE_LOGPROB_MAX
        segment_above_max.start = 0.0
        segment_above_max.end = 1.0
        
        confidence_above = transcriber._calculate_confidence([segment_above_max])
        assert confidence_above == 1.0  # Should clamp to 1.0



    def test_create_transcription_result_basic(self) -> None:
        """Test basic TranscriptionResult creation."""
        transcriber = WhisperTranscriber()
        
        result = transcriber.create_transcription_result(
            text="Hello world",
            processing_start_time=1000.0,
            confidence=0.95
        )
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.processing_time > 0.0

    def test_create_transcription_result_various_confidence_levels(self) -> None:
        """Test TranscriptionResult creation with various confidence levels."""
        transcriber = WhisperTranscriber()
        
        # Test different confidence levels - all should have default metadata
        test_cases = [
            ("High confidence", 0.95),
            ("Medium confidence", 0.65),
            ("Low confidence", 0.3),
            ("Very low confidence", 0.1)
        ]
        
        for text, confidence in test_cases:
            result = transcriber.create_transcription_result(
                text=text,
                processing_start_time=1000.0,
                confidence=confidence
            )
            
            assert isinstance(result, TranscriptionResult)
            assert result.text == text
            assert result.confidence == confidence
            assert result.processing_time > 0.0

    def test_confidence_calculation_uses_config_constants(self) -> None:
        """Test that confidence calculation uses configuration constants correctly."""
        transcriber = WhisperTranscriber()
        
        # Test with logprob at minimum expected value
        segment_min = Mock()
        segment_min.avg_logprob = CONFIDENCE_LOGPROB_MIN
        segment_min.start = 0.0
        segment_min.end = 1.0
        
        confidence_min = transcriber._calculate_confidence([segment_min])
        assert confidence_min == 0.0
        
        # Test with logprob at maximum expected value
        segment_max = Mock()
        segment_max.avg_logprob = CONFIDENCE_LOGPROB_MAX
        segment_max.start = 0.0
        segment_max.end = 1.0
        
        confidence_max = transcriber._calculate_confidence([segment_max])
        assert confidence_max == 1.0

    def test_confidence_metadata_integration(self) -> None:
        """Test that confidence metadata uses default values."""
        transcriber = WhisperTranscriber()
        
        # Test various confidence levels - all should have default metadata
        test_cases = [0.95, 0.65, 0.3, 0.1]
        
        for confidence in test_cases:
            result = transcriber.create_transcription_result(
                text=f"Test {confidence}",
                processing_start_time=1000.0,
                confidence=confidence
            )
            
            assert result.confidence == confidence