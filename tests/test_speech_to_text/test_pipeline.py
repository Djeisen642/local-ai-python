"""Tests for plugin-style processing pipeline."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from local_ai.speech_to_text.pipeline import (
    PluginProcessingPipeline,
    create_processing_context
)
from local_ai.speech_to_text.interfaces import (
    ProcessingStage,
    ProcessingContext,
    ProcessingResult,
    ProcessingHandler
)

# Import mock handlers from test_interfaces
from .test_interfaces import MockEmbeddingHandler, MockResponseHandler, MockTTSHandler


class MockFailingHandler(ProcessingHandler):
    """Mock handler that always fails for testing error handling."""
    
    def __init__(self, name: str = "failing_handler", stage: ProcessingStage = ProcessingStage.EMBEDDING) -> None:
        self._name = name
        self._stage = stage
    
    @property
    def stage(self) -> ProcessingStage:
        return self._stage
    
    @property
    def name(self) -> str:
        return self._name
    
    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Always raises an exception."""
        raise RuntimeError("Simulated handler failure")
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """Always returns True."""
        return True


class MockSlowHandler(ProcessingHandler):
    """Mock handler that takes time to process for testing concurrency."""
    
    def __init__(self, name: str = "slow_handler", delay: float = 0.1) -> None:
        self._name = name
        self._delay = delay
        self.process_called = False
    
    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.EMBEDDING
    
    @property
    def name(self) -> str:
        return self._name
    
    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Simulates slow processing."""
        self.process_called = True
        await asyncio.sleep(self._delay)
        
        return ProcessingResult(
            success=True,
            stage=self.stage,
            data={"processed": True, "delay": self._delay},
            processing_time=self._delay
        )
    
    def can_handle(self, context: ProcessingContext) -> bool:
        """Always returns True."""
        return True


@pytest.mark.unit
class TestPluginProcessingPipeline:
    """Test cases for PluginProcessingPipeline."""
    
    def test_pipeline_initialization(self) -> None:
        """Test pipeline initializes with correct default state."""
        pipeline = PluginProcessingPipeline()
        
        assert pipeline.get_registered_handlers() == []
        assert pipeline.get_handler_info() == {}
        
        stats = pipeline.get_pipeline_stats()
        assert stats["total_processed"] == 0
        assert stats["successful_processed"] == 0
        assert stats["failed_processed"] == 0
        assert stats["average_processing_time"] == 0.0
        assert stats["last_processed"] is None
    
    def test_handler_registration(self) -> None:
        """Test handler registration and unregistration."""
        pipeline = PluginProcessingPipeline()
        
        # Register handlers
        embedding_handler = MockEmbeddingHandler("test_embedding")
        response_handler = MockResponseHandler("test_response")
        tts_handler = MockTTSHandler("test_tts")
        
        assert pipeline.register_handler(embedding_handler) is True
        assert pipeline.register_handler(response_handler) is True
        assert pipeline.register_handler(tts_handler) is True
        
        # Check registered handlers
        all_handlers = pipeline.get_registered_handlers()
        assert len(all_handlers) == 3
        
        embedding_handlers = pipeline.get_registered_handlers(ProcessingStage.EMBEDDING)
        assert len(embedding_handlers) == 1
        assert embedding_handlers[0].name == "test_embedding"
        
        # Check handler info
        handler_info = pipeline.get_handler_info()
        assert "embedding" in handler_info
        assert "response_generation" in handler_info
        assert "text_to_speech" in handler_info
        assert "test_embedding" in handler_info["embedding"]
        assert "test_response" in handler_info["response_generation"]
        assert "test_tts" in handler_info["text_to_speech"]
    
    def test_handler_replacement(self) -> None:
        """Test that registering a handler with the same name replaces the existing one."""
        pipeline = PluginProcessingPipeline()
        
        # Register first handler
        handler1 = MockEmbeddingHandler("same_name")
        assert pipeline.register_handler(handler1) is True
        
        # Register second handler with same name
        handler2 = MockEmbeddingHandler("same_name")
        assert pipeline.register_handler(handler2) is True
        
        # Should only have one handler
        handlers = pipeline.get_registered_handlers(ProcessingStage.EMBEDDING)
        assert len(handlers) == 1
        assert handlers[0] is handler2  # Should be the second handler
    
    def test_handler_unregistration(self) -> None:
        """Test handler unregistration."""
        pipeline = PluginProcessingPipeline()
        
        # Register handler
        handler = MockEmbeddingHandler("test_handler")
        pipeline.register_handler(handler)
        
        # Verify registration
        assert len(pipeline.get_registered_handlers(ProcessingStage.EMBEDDING)) == 1
        
        # Unregister handler
        assert pipeline.unregister_handler(ProcessingStage.EMBEDDING, "test_handler") is True
        
        # Verify unregistration
        assert len(pipeline.get_registered_handlers(ProcessingStage.EMBEDDING)) == 0
        
        # Try to unregister non-existent handler
        assert pipeline.unregister_handler(ProcessingStage.EMBEDDING, "nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_pipeline_processing_success(self) -> None:
        """Test successful pipeline processing with multiple handlers."""
        pipeline = PluginProcessingPipeline()
        
        # Register handlers
        embedding_handler = MockEmbeddingHandler("test_embedding")
        response_handler = MockResponseHandler("test_response")
        tts_handler = MockTTSHandler("test_tts")
        
        pipeline.register_handler(embedding_handler)
        pipeline.register_handler(response_handler)
        pipeline.register_handler(tts_handler)
        
        # Create context
        context = create_processing_context(
            text="Can you help me?",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5
        )
        
        # Process through pipeline
        results = await pipeline.process_transcription(context)
        
        # Should have results from embedding, response, and TTS handlers
        assert len(results) == 3
        
        # Check that all handlers were called
        assert embedding_handler.generate_embedding_called is True
        assert response_handler.generate_response_called is True
        assert tts_handler.synthesize_speech_called is True
        
        # Check results
        embedding_result = next(r for r in results if r.stage == ProcessingStage.EMBEDDING)
        response_result = next(r for r in results if r.stage == ProcessingStage.RESPONSE_GENERATION)
        tts_result = next(r for r in results if r.stage == ProcessingStage.TEXT_TO_SPEECH)
        
        assert embedding_result.success is True
        assert response_result.success is True
        assert tts_result.success is True
        
        # Check pipeline stats
        stats = pipeline.get_pipeline_stats()
        assert stats["total_processed"] == 1
        assert stats["successful_processed"] == 1
        assert stats["failed_processed"] == 0
        assert stats["average_processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_processing_with_handler_filtering(self) -> None:
        """Test pipeline processing with handler filtering based on can_handle."""
        pipeline = PluginProcessingPipeline()
        
        # Register handlers
        embedding_handler = MockEmbeddingHandler("test_embedding")
        response_handler = MockResponseHandler("test_response")  # Only handles questions
        
        pipeline.register_handler(embedding_handler)
        pipeline.register_handler(response_handler)
        
        # Create context with statement (not a question)
        context = create_processing_context(
            text="This is a statement.",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5
        )
        
        # Process through pipeline
        results = await pipeline.process_transcription(context)
        
        # Should only have result from embedding handler (response handler can't handle statements)
        assert len(results) == 1
        assert results[0].stage == ProcessingStage.EMBEDDING
        
        # Check that only embedding handler was called
        assert embedding_handler.generate_embedding_called is True
        assert response_handler.generate_response_called is False
    
    @pytest.mark.asyncio
    async def test_pipeline_processing_with_failures(self) -> None:
        """Test pipeline processing with handler failures."""
        pipeline = PluginProcessingPipeline()
        
        # Register handlers including a failing one
        embedding_handler = MockEmbeddingHandler("test_embedding")
        failing_handler = MockFailingHandler("failing_handler", ProcessingStage.EMBEDDING)
        
        pipeline.register_handler(embedding_handler)
        pipeline.register_handler(failing_handler)
        
        # Create context
        context = create_processing_context(
            text="Test text",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5
        )
        
        # Process through pipeline
        results = await pipeline.process_transcription(context)
        
        # Should have results from both handlers (one success, one failure)
        assert len(results) == 2
        
        # Check results
        success_results = [r for r in results if r.success]
        failure_results = [r for r in results if not r.success]
        
        assert len(success_results) == 1
        assert len(failure_results) == 1
        
        # Check failure result
        failure_result = failure_results[0]
        assert failure_result.stage == ProcessingStage.EMBEDDING
        assert "Simulated handler failure" in failure_result.error
        
        # Check pipeline stats
        stats = pipeline.get_pipeline_stats()
        assert stats["total_processed"] == 1
        assert stats["successful_processed"] == 1  # Pipeline succeeded despite handler failure
        assert stats["failed_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(self) -> None:
        """Test that handlers within the same stage are processed concurrently."""
        pipeline = PluginProcessingPipeline()
        
        # Register multiple slow handlers for the same stage
        handler1 = MockSlowHandler("slow1", delay=0.1)
        handler2 = MockSlowHandler("slow2", delay=0.1)
        handler3 = MockSlowHandler("slow3", delay=0.1)
        
        pipeline.register_handler(handler1)
        pipeline.register_handler(handler2)
        pipeline.register_handler(handler3)
        
        # Create context
        context = create_processing_context(
            text="Test concurrent processing",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5
        )
        
        # Measure processing time
        import time
        start_time = time.time()
        results = await pipeline.process_transcription(context)
        end_time = time.time()
        
        # Should have results from all handlers
        assert len(results) == 3
        
        # Processing should take roughly the delay time (concurrent), not 3x delay time (sequential)
        processing_time = end_time - start_time
        assert processing_time < 0.25  # Should be much less than 0.3 (3 * 0.1) if concurrent
        
        # All handlers should have been called
        assert handler1.process_called is True
        assert handler2.process_called is True
        assert handler3.process_called is True
    
    @pytest.mark.asyncio
    async def test_empty_pipeline_processing(self) -> None:
        """Test processing with no registered handlers."""
        pipeline = PluginProcessingPipeline()
        
        # Create context
        context = create_processing_context(
            text="Test text",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5
        )
        
        # Process through empty pipeline
        results = await pipeline.process_transcription(context)
        
        # Should have no results
        assert len(results) == 0
        
        # Check pipeline stats
        stats = pipeline.get_pipeline_stats()
        assert stats["total_processed"] == 1
        assert stats["successful_processed"] == 1
        assert stats["failed_processed"] == 0
    
    def test_pipeline_stats_reset(self) -> None:
        """Test pipeline statistics reset functionality."""
        pipeline = PluginProcessingPipeline()
        
        # Manually update stats to simulate processing
        pipeline._pipeline_stats["total_processed"] = 10
        pipeline._pipeline_stats["successful_processed"] = 8
        pipeline._pipeline_stats["failed_processed"] = 2
        pipeline._pipeline_stats["average_processing_time"] = 0.5
        pipeline._pipeline_stats["last_processed"] = "2023-01-01T00:00:00"
        
        # Reset stats
        pipeline.reset_stats()
        
        # Check that stats are reset
        stats = pipeline.get_pipeline_stats()
        assert stats["total_processed"] == 0
        assert stats["successful_processed"] == 0
        assert stats["failed_processed"] == 0
        assert stats["average_processing_time"] == 0.0
        assert stats["last_processed"] is None


@pytest.mark.unit
class TestCreateProcessingContext:
    """Test cases for create_processing_context utility function."""
    
    def test_create_processing_context_with_all_parameters(self) -> None:
        """Test creating processing context with all parameters."""
        context = create_processing_context(
            text="Hello world",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.5,
            audio_duration=2.0,
            sample_rate=16000,
            chunk_count=5,
            session_id="test-session",
            user_id="test-user",
            metadata={"key": "value"}
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
        assert context.metadata == {"key": "value"}
    
    def test_create_processing_context_with_defaults(self) -> None:
        """Test creating processing context with default values."""
        context = create_processing_context(
            text="Test",
            confidence=0.8,
            timestamp=1234567890.0,
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            chunk_count=2
        )
        
        assert context.text == "Test"
        assert context.confidence == 0.8
        assert context.session_id is not None  # Should generate UUID
        assert context.user_id is None
        assert context.metadata == {}
        assert context.stage == ProcessingStage.TRANSCRIPTION
    
    def test_create_processing_context_generates_unique_session_ids(self) -> None:
        """Test that unique session IDs are generated when not provided."""
        context1 = create_processing_context(
            text="Test1", confidence=0.8, timestamp=1234567890.0,
            processing_time=0.1, audio_duration=1.0, sample_rate=16000, chunk_count=2
        )
        
        context2 = create_processing_context(
            text="Test2", confidence=0.8, timestamp=1234567890.0,
            processing_time=0.1, audio_duration=1.0, sample_rate=16000, chunk_count=2
        )
        
        assert context1.session_id != context2.session_id
        assert context1.session_id is not None
        assert context2.session_id is not None