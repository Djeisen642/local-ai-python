"""Speech-to-text module for local AI application."""

from .models import AudioChunk, SpeechSegment, TranscriptionResult
from .service import SpeechToTextService
from .interfaces import (
    ProcessingStage,
    ProcessingContext,
    ProcessingResult,
    ProcessingHandler,
    EmbeddingHandler,
    ResponseGenerationHandler,
    TextToSpeechHandler,
    ProcessingPipeline
)
from .pipeline import PluginProcessingPipeline, create_processing_context

__all__ = [
    "AudioChunk",
    "SpeechSegment",
    "TranscriptionResult",
    "SpeechToTextService",
    "ProcessingStage",
    "ProcessingContext",
    "ProcessingResult",
    "ProcessingHandler",
    "EmbeddingHandler",
    "ResponseGenerationHandler",
    "TextToSpeechHandler",
    "ProcessingPipeline",
    "PluginProcessingPipeline",
    "create_processing_context"
]
