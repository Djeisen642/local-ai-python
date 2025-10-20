"""Tests for service integration with processing pipeline."""

import asyncio

import pytest

from local_ai.speech_to_text.interfaces import ProcessingResult, ProcessingStage
from local_ai.speech_to_text.service import SpeechToTextService

from .test_interfaces import MockEmbeddingHandler, MockResponseHandler, MockTTSHandler


class TestServicePipelineIntegration:
    """Test cases for SpeechToTextService pipeline integration."""

    def test_service_pipeline_initialization(self) -> None:
        """Test that service initializes with processing pipeline."""
        service = SpeechToTextService()

        # Should have pipeline initialized
        assert service._processing_pipeline is not None
        assert service._pipeline_callback is None

        # Should have empty pipeline initially
        assert service.get_registered_handlers() == {}

        # Should have pipeline stats
        stats = service.get_pipeline_stats()
        assert "total_processed" in stats
        assert stats["total_processed"] == 0

    def test_handler_registration_and_unregistration(self) -> None:
        """Test handler registration and unregistration through service."""
        service = SpeechToTextService()

        # Register handlers
        embedding_handler = MockEmbeddingHandler("test_embedding")
        response_handler = MockResponseHandler("test_response")
        tts_handler = MockTTSHandler("test_tts")

        assert service.register_processing_handler(embedding_handler) is True
        assert service.register_processing_handler(response_handler) is True
        assert service.register_processing_handler(tts_handler) is True

        # Check registered handlers
        handlers = service.get_registered_handlers()
        assert "embedding" in handlers
        assert "response_generation" in handlers
        assert "text_to_speech" in handlers
        assert "test_embedding" in handlers["embedding"]
        assert "test_response" in handlers["response_generation"]
        assert "test_tts" in handlers["text_to_speech"]

        # Unregister handlers
        assert (
            service.unregister_processing_handler("embedding", "test_embedding") is True
        )
        assert (
            service.unregister_processing_handler("response_generation", "test_response")
            is True
        )
        assert service.unregister_processing_handler("text_to_speech", "test_tts") is True

        # Check handlers are unregistered
        handlers = service.get_registered_handlers()
        assert handlers.get("embedding", []) == []
        assert handlers.get("response_generation", []) == []
        assert handlers.get("text_to_speech", []) == []

    def test_invalid_stage_unregistration(self) -> None:
        """Test unregistration with invalid stage name."""
        service = SpeechToTextService()

        # Try to unregister with invalid stage
        assert service.unregister_processing_handler("invalid_stage", "handler") is False

    def test_pipeline_callback_mechanism(self) -> None:
        """Test pipeline callback mechanism."""
        service = SpeechToTextService()
        callback_results = []

        def pipeline_callback(results: list[ProcessingResult]) -> None:
            callback_results.append(results)

        # Set pipeline callback
        service.set_pipeline_callback(pipeline_callback)
        assert service._pipeline_callback is pipeline_callback

        # Test callback can be called
        test_results = [
            ProcessingResult(
                success=True, stage=ProcessingStage.EMBEDDING, data={"test": "data"}
            )
        ]

        if service._pipeline_callback:
            service._pipeline_callback(test_results)

        assert len(callback_results) == 1
        assert callback_results[0] == test_results

    @pytest.mark.asyncio
    async def test_transcription_triggers_pipeline(self) -> None:
        """Test that transcription updates trigger pipeline processing."""
        service = SpeechToTextService()

        # Register a mock handler
        embedding_handler = MockEmbeddingHandler("test_embedding")
        service.register_processing_handler(embedding_handler)

        # Set up pipeline callback to capture results
        pipeline_results = []

        def pipeline_callback(results: list[ProcessingResult]) -> None:
            pipeline_results.append(results)

        service.set_pipeline_callback(pipeline_callback)

        # Create transcription metadata
        metadata = {
            "confidence": 0.9,
            "timestamp": 1234567890.0,
            "processing_time": 0.5,
            "audio_duration": 2.0,
            "sample_rate": 16000,
            "chunk_count": 5,
            "session_id": "test-session",
            "user_id": "test-user",
            "additional_metadata": {"test": "data"},
        }

        # Trigger transcription update
        service._update_transcription("Test transcription", metadata)

        # Wait for async pipeline processing to complete
        await asyncio.sleep(0.1)

        # Check that pipeline was triggered
        assert len(pipeline_results) == 1
        results = pipeline_results[0]
        assert len(results) == 1
        assert results[0].stage == ProcessingStage.EMBEDDING
        assert results[0].success is True

        # Check that handler was called
        assert embedding_handler.generate_embedding_called is True
        assert embedding_handler.store_embedding_called is True

    @pytest.mark.asyncio
    async def test_transcription_without_metadata_skips_pipeline(self) -> None:
        """Test that transcription without metadata doesn't trigger pipeline."""
        service = SpeechToTextService()

        # Register a mock handler
        embedding_handler = MockEmbeddingHandler("test_embedding")
        service.register_processing_handler(embedding_handler)

        # Set up pipeline callback to capture results
        pipeline_results = []

        def pipeline_callback(results: list[ProcessingResult]) -> None:
            pipeline_results.append(results)

        service.set_pipeline_callback(pipeline_callback)

        # Trigger transcription update without metadata
        service._update_transcription("Test transcription")

        # Wait for potential async processing
        await asyncio.sleep(0.1)

        # Check that pipeline was not triggered
        assert len(pipeline_results) == 0
        assert embedding_handler.generate_embedding_called is False

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self) -> None:
        """Test error handling in pipeline processing."""
        service = SpeechToTextService()

        # Create a handler that will fail
        class FailingHandler(MockEmbeddingHandler):
            async def process(self, context):
                raise RuntimeError("Handler failure")

        failing_handler = FailingHandler("failing_handler")
        service.register_processing_handler(failing_handler)

        # Set up pipeline callback to capture results
        pipeline_results = []

        def pipeline_callback(results: list[ProcessingResult]) -> None:
            pipeline_results.append(results)

        service.set_pipeline_callback(pipeline_callback)

        # Create transcription metadata
        metadata = {
            "confidence": 0.9,
            "timestamp": 1234567890.0,
            "processing_time": 0.5,
            "audio_duration": 2.0,
            "sample_rate": 16000,
            "chunk_count": 5,
        }

        # Trigger transcription update
        service._update_transcription("Test transcription", metadata)

        # Wait for async pipeline processing to complete
        await asyncio.sleep(0.1)

        # Check that pipeline was triggered and error was handled
        assert len(pipeline_results) == 1
        results = pipeline_results[0]
        assert len(results) == 1
        assert results[0].stage == ProcessingStage.EMBEDDING
        assert results[0].success is False
        assert "Handler failure" in results[0].error

    @pytest.mark.asyncio
    async def test_pipeline_callback_error_handling(self) -> None:
        """Test error handling when pipeline callback fails."""
        service = SpeechToTextService()

        # Register a mock handler
        embedding_handler = MockEmbeddingHandler("test_embedding")
        service.register_processing_handler(embedding_handler)

        # Set up failing pipeline callback
        def failing_callback(results: list[ProcessingResult]) -> None:
            raise RuntimeError("Callback failure")

        service.set_pipeline_callback(failing_callback)

        # Create transcription metadata
        metadata = {
            "confidence": 0.9,
            "timestamp": 1234567890.0,
            "processing_time": 0.5,
            "audio_duration": 2.0,
            "sample_rate": 16000,
            "chunk_count": 5,
        }

        # Trigger transcription update - should not raise exception
        try:
            service._update_transcription("Test transcription", metadata)
            await asyncio.sleep(0.1)  # Wait for async processing
        except Exception:
            pytest.fail("Pipeline callback error should be handled gracefully")

        # Handler should still have been called
        assert embedding_handler.generate_embedding_called is True

    def test_pipeline_stats_access(self) -> None:
        """Test access to pipeline statistics through service."""
        service = SpeechToTextService()

        # Get initial stats
        stats = service.get_pipeline_stats()
        assert stats["total_processed"] == 0
        assert stats["successful_processed"] == 0
        assert stats["failed_processed"] == 0

        # Manually update pipeline stats to test access
        service._processing_pipeline._pipeline_stats["total_processed"] = 5
        service._processing_pipeline._pipeline_stats["successful_processed"] = 4
        service._processing_pipeline._pipeline_stats["failed_processed"] = 1

        # Check updated stats
        stats = service.get_pipeline_stats()
        assert stats["total_processed"] == 5
        assert stats["successful_processed"] == 4
        assert stats["failed_processed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_handlers_cascade_processing(self) -> None:
        """Test cascade processing with multiple handlers."""
        service = SpeechToTextService()

        # Register handlers for different stages
        embedding_handler = MockEmbeddingHandler("test_embedding")
        response_handler = MockResponseHandler("test_response")
        tts_handler = MockTTSHandler("test_tts")

        service.register_processing_handler(embedding_handler)
        service.register_processing_handler(response_handler)
        service.register_processing_handler(tts_handler)

        # Set up pipeline callback to capture results
        pipeline_results = []

        def pipeline_callback(results: list[ProcessingResult]) -> None:
            pipeline_results.append(results)

        service.set_pipeline_callback(pipeline_callback)

        # Create transcription metadata for a question (to trigger response handler)
        metadata = {
            "confidence": 0.9,
            "timestamp": 1234567890.0,
            "processing_time": 0.5,
            "audio_duration": 2.0,
            "sample_rate": 16000,
            "chunk_count": 5,
        }

        # Trigger transcription update with a question
        service._update_transcription("Can you help me?", metadata)

        # Wait for async pipeline processing to complete
        await asyncio.sleep(0.2)

        # Check that all handlers were triggered
        assert len(pipeline_results) == 1
        results = pipeline_results[0]

        # Should have results from all three stages
        stages = [result.stage for result in results]
        assert ProcessingStage.EMBEDDING in stages
        assert ProcessingStage.RESPONSE_GENERATION in stages
        assert ProcessingStage.TEXT_TO_SPEECH in stages

        # All results should be successful
        assert all(result.success for result in results)

        # Check that all handlers were called
        assert embedding_handler.generate_embedding_called is True
        assert response_handler.generate_response_called is True
        assert tts_handler.synthesize_speech_called is True

    @pytest.mark.asyncio
    async def test_context_metadata_preservation(self) -> None:
        """Test that processing context preserves all necessary metadata."""
        service = SpeechToTextService()

        # Create a custom handler to inspect the context
        received_contexts = []

        class InspectingHandler(MockEmbeddingHandler):
            async def process(self, context):
                received_contexts.append(context)
                return await super().process(context)

        inspecting_handler = InspectingHandler("inspecting_handler")
        service.register_processing_handler(inspecting_handler)

        # Create comprehensive transcription metadata
        metadata = {
            "confidence": 0.95,
            "timestamp": 1234567890.5,
            "processing_time": 0.75,
            "audio_duration": 3.5,
            "sample_rate": 22050,
            "chunk_count": 7,
            "session_id": "test-session-123",
            "user_id": "user-456",
            "additional_metadata": {
                "audio_size_bytes": 140000,
                "optimization_target": "accuracy",
                "monitoring_enabled": True,
                "custom_field": "custom_value",
            },
        }

        # Trigger transcription update
        service._update_transcription("Test transcription with metadata", metadata)

        # Wait for async pipeline processing to complete
        await asyncio.sleep(0.1)

        # Check that context was created with all metadata
        assert len(received_contexts) == 1
        context = received_contexts[0]

        assert context.text == "Test transcription with metadata"
        assert context.confidence == 0.95
        assert context.timestamp == 1234567890.5
        assert context.processing_time == 0.75
        assert context.audio_duration == 3.5
        assert context.sample_rate == 22050
        assert context.chunk_count == 7
        assert context.session_id == "test-session-123"
        assert context.user_id == "user-456"

        # Check additional metadata
        assert context.metadata["audio_size_bytes"] == 140000
        assert context.metadata["optimization_target"] == "accuracy"
        assert context.metadata["monitoring_enabled"] is True
        assert context.metadata["custom_field"] == "custom_value"
