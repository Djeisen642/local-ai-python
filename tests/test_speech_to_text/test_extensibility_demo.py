"""Demo test showing extensible interfaces for future system integration."""

import asyncio
from typing import Any

import pytest

from local_ai.speech_to_text.interfaces import (
    EmbeddingHandler,
    ProcessingContext,
    ProcessingResult,
    ProcessingStage,
    ResponseGenerationHandler,
    TextToSpeechHandler,
)
from local_ai.speech_to_text.service import SpeechToTextService


class DemoEmbeddingHandler(EmbeddingHandler):
    """Demo embedding handler that simulates semantic embedding generation."""

    def __init__(self) -> None:
        self.embeddings_store = {}  # Simple in-memory store

    @property
    def name(self) -> str:
        return "demo_semantic_embeddings"

    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process transcription to generate and store embeddings."""
        try:
            # Generate embedding
            embedding = await self.generate_embedding(context.text, context.metadata)

            # Store embedding with context
            stored = await self.store_embedding(embedding, context)

            return ProcessingResult(
                success=stored,
                stage=self.stage,
                data={
                    "embedding_vector": embedding,
                    "embedding_id": f"emb_{context.session_id}_{len(self.embeddings_store)}",
                    "stored": stored,
                },
                processing_time=0.05,
            )
        except Exception as e:
            return ProcessingResult(
                success=False, stage=self.stage, data=None, error=str(e)
            )

    def can_handle(self, context: ProcessingContext) -> bool:
        """Can handle any non-empty text."""
        return len(context.text.strip()) > 0

    async def generate_embedding(
        self, text: str, metadata: dict[str, Any]
    ) -> list[float]:
        """Generate semantic embedding for text (simulated)."""
        # Simulate embedding generation based on text characteristics
        text_length = len(text)
        word_count = len(text.split())

        # Create a simple embedding based on text features
        return [
            text_length / 100.0,  # Normalized length
            word_count / 20.0,  # Normalized word count
            1.0 if "?" in text else 0.0,  # Question indicator
            1.0
            if any(word in text.lower() for word in ["help", "please", "thank"])
            else 0.0,  # Politeness
            metadata.get("confidence", 0.0),  # Transcription confidence
        ]

    async def store_embedding(
        self, embedding: list[float], context: ProcessingContext
    ) -> bool:
        """Store embedding with context for future retrieval."""
        embedding_id = f"emb_{context.session_id}_{len(self.embeddings_store)}"

        self.embeddings_store[embedding_id] = {
            "embedding": embedding,
            "text": context.text,
            "timestamp": context.timestamp,
            "session_id": context.session_id,
            "confidence": context.confidence,
            "metadata": context.metadata,
        }

        return True

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for similar embeddings (simple cosine similarity)."""

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

        similarities = []
        for emb_id, data in self.embeddings_store.items():
            similarity = cosine_similarity(query_embedding, data["embedding"])
            similarities.append((similarity, emb_id, data))

        # Sort by similarity and return top_k
        similarities.sort(reverse=True)
        return [
            {"id": emb_id, "similarity": sim, "data": data}
            for sim, emb_id, data in similarities[:top_k]
        ]


class DemoResponseHandler(ResponseGenerationHandler):
    """Demo response handler that simulates AI response generation."""

    def __init__(self, embedding_handler: DemoEmbeddingHandler) -> None:
        self.embedding_handler = embedding_handler
        self.conversation_history = {}

    @property
    def name(self) -> str:
        return "demo_ai_response"

    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process transcription to generate AI response."""
        try:
            # Generate response
            response = await self.generate_response(context.text, context)

            # Store in conversation history
            self._add_to_history(context.session_id, "user", context.text)
            self._add_to_history(context.session_id, "assistant", response)

            return ProcessingResult(
                success=True,
                stage=self.stage,
                data={
                    "response": response,
                    "conversation_turn": len(
                        self.conversation_history.get(context.session_id, [])
                    ),
                    "response_type": self._classify_response_type(context.text),
                },
                processing_time=0.1,
            )
        except Exception as e:
            return ProcessingResult(
                success=False, stage=self.stage, data=None, error=str(e)
            )

    def can_handle(self, context: ProcessingContext) -> bool:
        """Can handle questions and requests for help."""
        text_lower = context.text.lower()
        return "?" in context.text or any(
            word in text_lower for word in ["help", "what", "how", "why", "when", "where"]
        )

    async def generate_response(self, text: str, context: ProcessingContext) -> str:
        """Generate AI response based on input text and context."""
        # Get conversation context
        await self.get_conversation_context(context.session_id)

        # Search for similar past interactions
        if self.embedding_handler.embeddings_store:
            query_embedding = await self.embedding_handler.generate_embedding(
                text, context.metadata
            )
            similar = self.embedding_handler.search_similar(query_embedding, top_k=3)
        else:
            similar = []

        # Generate contextual response
        response_type = self._classify_response_type(text)

        if response_type == "greeting":
            return "Hello! How can I help you today?"
        if response_type == "question":
            if similar:
                return f"Based on similar questions, I think you're asking about {text.lower()}. Let me help you with that."
            return (
                f"That's an interesting question about '{text}'. Let me think about that."
            )
        if response_type == "help_request":
            return "I'm here to help! What specific assistance do you need?"
        if response_type == "gratitude":
            return "You're welcome! Is there anything else I can help you with?"
        return f"I understand you said: '{text}'. How can I assist you further?"

    async def get_conversation_context(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve conversation context for the session."""
        return self.conversation_history.get(session_id, [])

    def _add_to_history(self, session_id: str, role: str, content: str) -> None:
        """Add message to conversation history."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append(
            {
                "role": role,
                "content": content,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def _classify_response_type(self, text: str) -> str:
        """Classify the type of response needed."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        if "?" in text or any(
            word in text_lower for word in ["what", "how", "why", "when", "where"]
        ):
            return "question"
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "help_request"
        if any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
            return "gratitude"
        return "general"


class DemoTTSHandler(TextToSpeechHandler):
    """Demo text-to-speech handler that simulates speech synthesis."""

    def __init__(self) -> None:
        self.synthesis_history = []
        self.voice_config = {
            "voice": "neural_voice_1",
            "speed": 1.0,
            "pitch": 0.0,
            "volume": 0.8,
        }

    @property
    def name(self) -> str:
        return "demo_neural_tts"

    async def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process text to generate and play speech."""
        try:
            # Synthesize speech
            audio_data = await self.synthesize_speech(context.text, self.voice_config)

            # Play audio (simulated)
            played = await self.play_audio(audio_data)

            return ProcessingResult(
                success=played,
                stage=self.stage,
                data={
                    "audio_generated": True,
                    "audio_size": len(audio_data),
                    "voice_config": self.voice_config,
                    "played": played,
                },
                processing_time=0.2,
            )
        except Exception as e:
            return ProcessingResult(
                success=False, stage=self.stage, data=None, error=str(e)
            )

    def can_handle(self, context: ProcessingContext) -> bool:
        """Can handle any text for speech synthesis."""
        return len(context.text.strip()) > 0

    async def synthesize_speech(self, text: str, voice_config: dict[str, Any]) -> bytes:
        """Synthesize speech from text (simulated)."""
        # Simulate speech synthesis
        text_length = len(text)
        estimated_duration = text_length * 0.1  # Rough estimate: 0.1s per character

        # Create mock audio data (in real implementation, this would be actual audio)
        audio_data = f"AUDIO_DATA:{text}:DURATION:{estimated_duration}:CONFIG:{voice_config}".encode()

        # Store synthesis record
        self.synthesis_history.append(
            {
                "text": text,
                "config": voice_config,
                "audio_size": len(audio_data),
                "duration": estimated_duration,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        return audio_data

    async def play_audio(self, audio_data: bytes) -> bool:
        """Play the generated audio (simulated)."""
        # Simulate audio playback
        audio_data.decode()

        # In real implementation, this would play the audio through speakers
        # For demo, we just simulate successful playback
        return True

    def get_synthesis_history(self) -> list[dict[str, Any]]:
        """Get history of speech synthesis operations."""
        return self.synthesis_history.copy()


class TestExtensibilityDemo:
    """Demo test showing complete extensible system integration."""

    @pytest.mark.asyncio
    async def test_complete_ai_pipeline_demo(self) -> None:
        """
        Demo test showing complete AI pipeline with embedding, response generation, and TTS.

        This test demonstrates:
        1. Speech-to-text transcription triggers downstream processing
        2. Embedding generation and storage for semantic search
        3. AI response generation with conversation context
        4. Text-to-speech synthesis for voice output
        5. Complete cascade processing through all stages
        """
        # Initialize service
        service = SpeechToTextService()

        # Create and register handlers
        embedding_handler = DemoEmbeddingHandler()
        response_handler = DemoResponseHandler(embedding_handler)
        tts_handler = DemoTTSHandler()

        service.register_processing_handler(embedding_handler)
        service.register_processing_handler(response_handler)
        service.register_processing_handler(tts_handler)

        # Set up pipeline callback to capture results
        pipeline_results = []

        def pipeline_callback(results: list[ProcessingResult]) -> None:
            pipeline_results.append(results)

        service.set_pipeline_callback(pipeline_callback)

        # Simulate a series of user interactions
        interactions = [
            "Hello, can you help me?",
            "What is machine learning?",
            "How does speech recognition work?",
            "Thank you for your help!",
        ]

        session_id = "demo-session-123"

        for i, user_input in enumerate(interactions):
            # Create transcription metadata
            metadata = {
                "confidence": 0.9 + (i * 0.02),  # Slightly increasing confidence
                "timestamp": 1234567890.0 + (i * 10),
                "processing_time": 0.5,
                "audio_duration": len(user_input) * 0.1,
                "sample_rate": 16000,
                "chunk_count": 3 + i,
                "session_id": session_id,
                "user_id": "demo-user",
                "additional_metadata": {
                    "interaction_number": i + 1,
                    "total_interactions": len(interactions),
                },
            }

            # Trigger transcription processing
            service._update_transcription(user_input, metadata)

            # Wait for async processing
            await asyncio.sleep(0.1)

        # Verify pipeline processing
        assert len(pipeline_results) == len(interactions)

        # Check that all interactions were processed
        for i, results in enumerate(pipeline_results):
            # Each interaction should trigger all three stages
            stages = [result.stage for result in results]
            assert ProcessingStage.EMBEDDING in stages
            assert ProcessingStage.RESPONSE_GENERATION in stages
            assert ProcessingStage.TEXT_TO_SPEECH in stages

            # All results should be successful
            assert all(result.success for result in results)

        # Verify embedding storage
        assert len(embedding_handler.embeddings_store) == len(interactions)

        # Verify conversation history
        conversation = await response_handler.get_conversation_context(session_id)
        assert len(conversation) == len(interactions) * 2  # User + assistant messages

        # Verify TTS synthesis
        synthesis_history = tts_handler.get_synthesis_history()
        assert len(synthesis_history) == len(interactions)

        # Test semantic search functionality
        query_embedding = await embedding_handler.generate_embedding("help with AI", {})
        similar_results = embedding_handler.search_similar(query_embedding, top_k=2)
        assert len(similar_results) > 0
        assert all("similarity" in result for result in similar_results)

        for i, (user_input, results) in enumerate(zip(interactions, pipeline_results)):
            for result in results:
                if result.stage == ProcessingStage.EMBEDDING:
                    pass

                elif result.stage == ProcessingStage.RESPONSE_GENERATION:
                    pass

                elif result.stage == ProcessingStage.TEXT_TO_SPEECH:
                    pass

        # Verify pipeline statistics
        stats = service.get_pipeline_stats()
        assert stats["total_processed"] == len(interactions)
        assert stats["successful_processed"] == len(interactions)
        assert stats["failed_processed"] == 0
        assert stats["average_processing_time"] > 0

    @pytest.mark.asyncio
    async def test_handler_filtering_and_selective_processing(self) -> None:
        """Test that handlers can selectively process based on content."""
        service = SpeechToTextService()

        # Register handlers
        embedding_handler = DemoEmbeddingHandler()
        response_handler = DemoResponseHandler(
            embedding_handler
        )  # Only handles questions
        tts_handler = DemoTTSHandler()

        service.register_processing_handler(embedding_handler)
        service.register_processing_handler(response_handler)
        service.register_processing_handler(tts_handler)

        # Set up pipeline callback
        pipeline_results = []
        service.set_pipeline_callback(lambda results: pipeline_results.append(results))

        # Test with statement (not a question) - should not trigger response handler
        metadata = {
            "confidence": 0.9,
            "timestamp": 1234567890.0,
            "processing_time": 0.5,
            "audio_duration": 2.0,
            "sample_rate": 16000,
            "chunk_count": 3,
            "session_id": "test-session",
        }

        service._update_transcription("This is just a statement.", metadata)
        await asyncio.sleep(0.1)

        # Should have embedding and TTS results, but no response generation
        results = pipeline_results[0]
        stages = [result.stage for result in results]

        assert ProcessingStage.EMBEDDING in stages
        assert ProcessingStage.RESPONSE_GENERATION not in stages  # Filtered out
        assert ProcessingStage.TEXT_TO_SPEECH in stages

        # Test with question - should trigger all handlers
        pipeline_results.clear()
        service._update_transcription("What is the weather like?", metadata)
        await asyncio.sleep(0.1)

        results = pipeline_results[0]
        stages = [result.stage for result in results]

        assert ProcessingStage.EMBEDDING in stages
        assert ProcessingStage.RESPONSE_GENERATION in stages  # Now included
        assert ProcessingStage.TEXT_TO_SPEECH in stages

    def test_handler_registration_and_management(self) -> None:
        """Test handler registration and management through service."""
        service = SpeechToTextService()

        # Initially no handlers
        assert service.get_registered_handlers() == {}

        # Register handlers
        embedding_handler = DemoEmbeddingHandler()
        response_handler = DemoResponseHandler(embedding_handler)
        tts_handler = DemoTTSHandler()

        assert service.register_processing_handler(embedding_handler) is True
        assert service.register_processing_handler(response_handler) is True
        assert service.register_processing_handler(tts_handler) is True

        # Check registration
        handlers = service.get_registered_handlers()
        assert "embedding" in handlers
        assert "response_generation" in handlers
        assert "text_to_speech" in handlers

        # Unregister handlers
        assert (
            service.unregister_processing_handler("embedding", "demo_semantic_embeddings")
            is True
        )
        assert (
            service.unregister_processing_handler(
                "response_generation", "demo_ai_response"
            )
            is True
        )
        assert (
            service.unregister_processing_handler("text_to_speech", "demo_neural_tts")
            is True
        )

        # Check unregistration
        handlers = service.get_registered_handlers()
        assert handlers.get("embedding", []) == []
        assert handlers.get("response_generation", []) == []
        assert handlers.get("text_to_speech", []) == []
