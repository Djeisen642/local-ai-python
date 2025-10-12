"""Speech-to-text module for local AI application."""

from .models import AudioChunk, SpeechSegment, TranscriptionResult
from .service import SpeechToTextService

__all__ = [
    "AudioChunk",
    "SpeechSegment",
    "TranscriptionResult",
    "SpeechToTextService",
]
