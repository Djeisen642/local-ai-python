"""Tests for speech-to-text data models."""

import pytest
from local_ai.speech_to_text.models import AudioChunk, SpeechSegment, TranscriptionResult


class TestAudioChunk:
    """Test cases for AudioChunk data model."""
    
    def test_audio_chunk_creation(self) -> None:
        """Test AudioChunk can be created with required fields."""
        chunk = AudioChunk(
            data=b"test_audio_data",
            timestamp=1234567890.0,
            sample_rate=16000,
            duration=1.0
        )
        
        assert chunk.data == b"test_audio_data"
        assert chunk.timestamp == 1234567890.0
        assert chunk.sample_rate == 16000
        assert chunk.duration == 1.0


class TestTranscriptionResult:
    """Test cases for TranscriptionResult data model."""
    
    def test_transcription_result_creation(self) -> None:
        """Test TranscriptionResult can be created with required fields."""
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.5
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.timestamp == 1234567890.0
        assert result.processing_time == 0.5


class TestSpeechSegment:
    """Test cases for SpeechSegment data model."""
    
    def test_speech_segment_creation(self) -> None:
        """Test SpeechSegment can be created with required fields."""
        segment = SpeechSegment(
            audio_data=b"speech_data",
            start_time=1.0,
            end_time=3.0,
            is_complete=True
        )
        
        assert segment.audio_data == b"speech_data"
        assert segment.start_time == 1.0
        assert segment.end_time == 3.0
        assert segment.is_complete is True