"""Abstract interfaces for future system integration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ProcessingStage(Enum):
    """Enumeration of processing stages in the AI pipeline."""
    TRANSCRIPTION = "transcription"
    EMBEDDING = "embedding"
    RESPONSE_GENERATION = "response_generation"
    TEXT_TO_SPEECH = "text_to_speech"


@dataclass
class ProcessingContext:
    """Context information passed between processing stages."""
    
    # Core transcription data
    text: str
    confidence: float
    timestamp: float
    processing_time: float
    
    # Audio metadata
    audio_duration: float
    sample_rate: int
    chunk_count: int
    
    # Processing metadata
    stage: ProcessingStage
    session_id: str
    user_id: Optional[str] = None
    
    # Extensible metadata for future systems
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata for downstream processing."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default."""
        return self.metadata.get(key, default)


@dataclass
class ProcessingResult:
    """Result from a processing stage."""
    
    success: bool
    stage: ProcessingStage
    data: Any
    error: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}


class ProcessingHandler(ABC):
    """Abstract base class for processing handlers in the AI pipeline."""
    
    @property
    @abstractmethod
    def stage(self) -> ProcessingStage:
        """Return the processing stage this handler manages."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique name for this handler."""
        pass
    
    @abstractmethod
    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        Process the given context and return a result.
        
        Args:
            context: Processing context with transcription and metadata
            
        Returns:
            ProcessingResult with success status and any output data
        """
        pass
    
    @abstractmethod
    def can_handle(self, context: ProcessingContext) -> bool:
        """
        Check if this handler can process the given context.
        
        Args:
            context: Processing context to evaluate
            
        Returns:
            True if this handler can process the context
        """
        pass
    
    def get_dependencies(self) -> List[ProcessingStage]:
        """
        Return list of processing stages this handler depends on.
        
        Returns:
            List of required processing stages
        """
        return []


class EmbeddingHandler(ProcessingHandler):
    """Abstract handler for text embedding generation."""
    
    @property
    def stage(self) -> ProcessingStage:
        """Return the embedding processing stage."""
        return ProcessingStage.EMBEDDING
    
    @abstractmethod
    async def generate_embedding(self, text: str, metadata: Dict[str, Any]) -> List[float]:
        """
        Generate embedding vector for the given text.
        
        Args:
            text: Text to generate embedding for
            metadata: Additional context metadata
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    async def store_embedding(self, embedding: List[float], context: ProcessingContext) -> bool:
        """
        Store embedding with associated context for retrieval.
        
        Args:
            embedding: Embedding vector to store
            context: Processing context with metadata
            
        Returns:
            True if storage was successful
        """
        pass


class ResponseGenerationHandler(ProcessingHandler):
    """Abstract handler for AI response generation."""
    
    @property
    def stage(self) -> ProcessingStage:
        """Return the response generation processing stage."""
        return ProcessingStage.RESPONSE_GENERATION
    
    @abstractmethod
    async def generate_response(self, text: str, context: ProcessingContext) -> str:
        """
        Generate AI response for the given input text.
        
        Args:
            text: Input text to respond to
            context: Processing context with conversation history
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    async def get_conversation_context(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation context for the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns with metadata
        """
        pass


class TextToSpeechHandler(ProcessingHandler):
    """Abstract handler for text-to-speech conversion."""
    
    @property
    def stage(self) -> ProcessingStage:
        """Return the text-to-speech processing stage."""
        return ProcessingStage.TEXT_TO_SPEECH
    
    @abstractmethod
    async def synthesize_speech(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice_config: Voice configuration parameters
            
        Returns:
            Audio data as bytes
        """
        pass
    
    @abstractmethod
    async def play_audio(self, audio_data: bytes) -> bool:
        """
        Play the generated audio.
        
        Args:
            audio_data: Audio data to play
            
        Returns:
            True if playback was successful
        """
        pass


class ProcessingPipeline(ABC):
    """Abstract interface for managing the processing pipeline."""
    
    @abstractmethod
    def register_handler(self, handler: ProcessingHandler) -> bool:
        """
        Register a processing handler.
        
        Args:
            handler: Handler to register
            
        Returns:
            True if registration was successful
        """
        pass
    
    @abstractmethod
    def unregister_handler(self, stage: ProcessingStage, name: str) -> bool:
        """
        Unregister a processing handler.
        
        Args:
            stage: Processing stage
            name: Handler name
            
        Returns:
            True if unregistration was successful
        """
        pass
    
    @abstractmethod
    async def process_transcription(self, context: ProcessingContext) -> List[ProcessingResult]:
        """
        Process transcription through the entire pipeline.
        
        Args:
            context: Processing context with transcription data
            
        Returns:
            List of results from each processing stage
        """
        pass
    
    @abstractmethod
    def get_registered_handlers(self, stage: Optional[ProcessingStage] = None) -> List[ProcessingHandler]:
        """
        Get list of registered handlers.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of registered handlers
        """
        pass