"""Tests for extensible interfaces for future system integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from local_ai.speech_to_text.interfaces import (
    ProcessingStage,
    ProcessingContext,
    ProcessingResult,
    ProcessingHandler,
    EmbeddingHandler,
    ResponseGenerationHandler,
    TextToSpeechHandler,
    ProcessingPipeline
)


class TestProcessingContext:
    """Test cases for ProcessingContext data model."""
    
    def test_processing_context_initialization(self) -> None:
        """Test ProcessingContext initializes with correct values."""
        context = ProcessingContext(
            text="Hello world",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.5,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test-session",
            user_id="test-user"
        )
        
        assert context.text == "Hello world"
        assert context.confidence == 0.95
        assert context.timestamp == 1234567890.0
        assert context.processing_time == 0.5
        assert context.audio_duration == 2.0
        assert context.sample_rate == 16000
        assert context.chunk_count == 5
        assert context.stage == ProcessingStage.TRANSCRIPTION
        assert context.session_id == "test-session"
        assert context.user_id == "test-user"
        assert context.metadata == {}
    
    def test_processing_context_metadata_operations(self) -> None:
        """Test metadata operations on ProcessingContext."""
        context = ProcessingContext(
            text="Test",
            confidence=0.8,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.EMBEDDING,
            session_id="test"
        )
        
        # Test adding metadata
        context.add_metadata("key1", "value1")
        context.add_metadata("key2", 42)
        
        assert context.get_metadata("key1") == "value1"
        assert context.get_metadata("key2") == 42
        assert context.get_metadata("nonexistent", "default") == "default"
        
        # Test metadata dict is properly initialized
        assert "key1" in context.metadata
        assert "key2" in context.metadata


class TestProcessingResult:
    """Test cases for ProcessingResult data model."""
    
    def test_processing_result_initialization(self) -> None:
        """Test ProcessingResult initializes with correct values."""
        result = ProcessingResult(
            success=True,
            stage=ProcessingStage.EMBEDDING,
            data={"embedding": [0.1, 0.2, 0.3]},
            processing_time=0.25
        )
        
        assert result.success is True
        assert result.stage == ProcessingStage.EMBEDDING
        assert result.data == {"embedding": [0.1, 0.2, 0.3]}
        assert result.error is None
        assert result.processing_time == 0.25
        assert result.metadata == {}
    
    def test_processing_result_with_error(self) -> None:
        """Test ProcessingResult with error information."""
        result = ProcessingResult(
            success=False,
            stage=ProcessingStage.RESPONSE_GENERATION,
            data=None,
            error="Model not available",
            processing_time=0.1
        )
        
        assert result.success is False
        assert result.stage == ProcessingStage.RESPONSE_GENERATION
        assert result.data is None
        assert result.error == "Model not available"
        assert result.processing_time == 0.1


class MockEmbeddingHandler(EmbeddingHandler):
    """Mock embedding handler for testing."""
    
    def __init__(self, name: str = "mock_embedding") -> None:
        self._name = name
        self.generate_embedding_called = False
        self.store_embedding_called = False
    
    @property
    def name(self) -> str:
        return self._name
    
    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Mock process method."""
        embedding = await self.generate_embedding(context.text, context.metadata)
        stored = await self.store_embedding(embedding, context)
        
        return ProcessingResult(
            success=stored,
            stage=self.stage,
            data={"embedding": embedding, "stored": stored},
            processing_time=0.1
        )
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """Mock can_handle method."""
        return len(context.text.strip()) > 0
    
    async def generate_embedding(self, text: str, metadata: Dict[str, Any]) -> List[float]:
        """Mock embedding generation."""
        self.generate_embedding_called = True
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    async def store_embedding(self, embedding: List[float], context: ProcessingContext) -> bool:
        """Mock embedding storage."""
        self.store_embedding_called = True
        return True


class MockResponseHandler(ResponseGenerationHandler):
    """Mock response generation handler for testing."""
    
    def __init__(self, name: str = "mock_response") -> None:
        self._name = name
        self.generate_response_called = False
        self.get_conversation_context_called = False
    
    @property
    def name(self) -> str:
        return self._name
    
    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Mock process method."""
        response = await self.generate_response(context.text, context)
        
        return ProcessingResult(
            success=True,
            stage=self.stage,
            data={"response": response},
            processing_time=0.2
        )
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """Mock can_handle method."""
        return "?" in context.text or "help" in context.text.lower()
    
    async def generate_response(self, text: str, context: ProcessingContext) -> str:
        """Mock response generation."""
        self.generate_response_called = True
        return f"Response to: {text}"
    
    async def get_conversation_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Mock conversation context retrieval."""
        self.get_conversation_context_called = True
        return [{"role": "user", "content": "Previous message"}]


class MockTTSHandler(TextToSpeechHandler):
    """Mock text-to-speech handler for testing."""
    
    def __init__(self, name: str = "mock_tts") -> None:
        self._name = name
        self.synthesize_speech_called = False
        self.play_audio_called = False
    
    @property
    def name(self) -> str:
        return self._name
    
    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Mock process method."""
        audio_data = await self.synthesize_speech(context.text, {})
        played = await self.play_audio(audio_data)
        
        return ProcessingResult(
            success=played,
            stage=self.stage,
            data={"audio_generated": True, "played": played},
            processing_time=0.3
        )
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """Mock can_handle method."""
        return True  # Always can handle
    
    async def synthesize_speech(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """Mock speech synthesis."""
        self.synthesize_speech_called = True
        return b"mock_audio_data"
    
    async def play_audio(self, audio_data: bytes) -> bool:
        """Mock audio playback."""
        self.play_audio_called = True
        return True


class TestProcessingHandlers:
    """Test cases for processing handler interfaces."""
    
    def test_embedding_handler_interface(self) -> None:
        """Test EmbeddingHandler interface implementation."""
        handler = MockEmbeddingHandler("test_embedding")
        
        assert handler.name == "test_embedding"
        assert handler.stage == ProcessingStage.EMBEDDING
        assert handler.get_dependencies() == []
    
    def test_response_handler_interface(self) -> None:
        """Test ResponseGenerationHandler interface implementation."""
        handler = MockResponseHandler("test_response")
        
        assert handler.name == "test_response"
        assert handler.stage == ProcessingStage.RESPONSE_GENERATION
        assert handler.get_dependencies() == []
    
    def test_tts_handler_interface(self) -> None:
        """Test TextToSpeechHandler interface implementation."""
        handler = MockTTSHandler("test_tts")
        
        assert handler.name == "test_tts"
        assert handler.stage == ProcessingStage.TEXT_TO_SPEECH
        assert handler.get_dependencies() == []
    
    @pytest.mark.asyncio
    async def test_embedding_handler_processing(self) -> None:
        """Test embedding handler processing functionality."""
        handler = MockEmbeddingHandler()
        context = ProcessingContext(
            text="Test embedding",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test"
        )
        
        # Test can_handle
        assert handler.can_handle(context) is True
        
        # Test processing
        result = await handler.process(context)
        
        assert result.success is True
        assert result.stage == ProcessingStage.EMBEDDING
        assert "embedding" in result.data
        assert "stored" in result.data
        assert handler.generate_embedding_called is True
        assert handler.store_embedding_called is True
    
    @pytest.mark.asyncio
    async def test_response_handler_processing(self) -> None:
        """Test response generation handler processing functionality."""
        handler = MockResponseHandler()
        context = ProcessingContext(
            text="Can you help me?",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test"
        )
        
        # Test can_handle
        assert handler.can_handle(context) is True
        
        # Test processing
        result = await handler.process(context)
        
        assert result.success is True
        assert result.stage == ProcessingStage.RESPONSE_GENERATION
        assert "response" in result.data
        assert handler.generate_response_called is True
    
    @pytest.mark.asyncio
    async def test_tts_handler_processing(self) -> None:
        """Test text-to-speech handler processing functionality."""
        handler = MockTTSHandler()
        context = ProcessingContext(
            text="Hello world",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test"
        )
        
        # Test can_handle
        assert handler.can_handle(context) is True
        
        # Test processing
        result = await handler.process(context)
        
        assert result.success is True
        assert result.stage == ProcessingStage.TEXT_TO_SPEECH
        assert "audio_generated" in result.data
        assert "played" in result.data
        assert handler.synthesize_speech_called is True
        assert handler.play_audio_called is True
    
    def test_handler_can_handle_filtering(self) -> None:
        """Test handler can_handle filtering logic."""
        embedding_handler = MockEmbeddingHandler()
        response_handler = MockResponseHandler()
        
        # Test embedding handler with empty text
        empty_context = ProcessingContext(
            text="   ",  # Only whitespace
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test"
        )
        
        assert embedding_handler.can_handle(empty_context) is False
        
        # Test response handler with non-question text
        statement_context = ProcessingContext(
            text="This is a statement",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test"
        )
        
        assert response_handler.can_handle(statement_context) is False
        
        # Test response handler with question
        question_context = ProcessingContext(
            text="What is the weather?",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2,
            stage=ProcessingStage.TRANSCRIPTION,
            session_id="test"
        )
        
        assert response_handler.can_handle(question_context) is True