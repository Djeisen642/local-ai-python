"""Speech-to-text module for local AI application."""

from .interfaces import (
    EmbeddingHandler,
    ProcessingContext,
    ProcessingHandler,
    ProcessingPipeline,
    ProcessingResult,
    ProcessingStage,
    ResponseGenerationHandler,
    TextToSpeechHandler,
)
from .models import AudioChunk, SpeechSegment, TranscriptionResult
from .pipeline import PluginProcessingPipeline, create_processing_context
from .service import SpeechToTextService

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
    "create_processing_context",
]
