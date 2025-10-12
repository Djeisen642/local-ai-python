# Extensible Interfaces for Future System Integration

The speech-to-text module provides extensible interfaces that allow future systems to integrate seamlessly with the transcription pipeline. This enables building complete AI assistant workflows with embedding generation, response generation, and text-to-speech capabilities.

## Overview

The extensible system consists of:

1. **Abstract Interfaces** - Define contracts for different processing stages
2. **Plugin Pipeline** - Manages registration and execution of processing handlers
3. **Service Integration** - Automatically triggers downstream processing when transcription completes
4. **Rich Metadata** - Preserves all necessary context for downstream systems

## Core Interfaces

### ProcessingStage

Defines the stages in the AI pipeline:

```python
from local_ai.speech_to_text import ProcessingStage

# Available stages
ProcessingStage.TRANSCRIPTION    # Speech-to-text conversion
ProcessingStage.EMBEDDING        # Semantic embedding generation
ProcessingStage.RESPONSE_GENERATION  # AI response generation
ProcessingStage.TEXT_TO_SPEECH   # Text-to-speech synthesis
```

### ProcessingContext

Contains all data and metadata passed between processing stages:

```python
from local_ai.speech_to_text import ProcessingContext

context = ProcessingContext(
    text="Hello, how are you?",
    confidence=0.95,
    timestamp=1234567890.0,
    processing_time=0.5,
    audio_duration=2.0,
    sample_rate=16000,
    chunk_count=5,
    stage=ProcessingStage.TRANSCRIPTION,
    session_id="session-123",
    user_id="user-456",
    metadata={"custom": "data"}
)

# Add additional metadata
context.add_metadata("key", "value")
value = context.get_metadata("key", default="default")
```

### ProcessingResult

Represents the output from a processing stage:

```python
from local_ai.speech_to_text import ProcessingResult, ProcessingStage

result = ProcessingResult(
    success=True,
    stage=ProcessingStage.EMBEDDING,
    data={"embedding": [0.1, 0.2, 0.3]},
    processing_time=0.1,
    metadata={"model": "sentence-transformer"}
)
```

## Creating Custom Handlers

### Embedding Handler

Create handlers for semantic embedding generation:

```python
from local_ai.speech_to_text import EmbeddingHandler, ProcessingContext, ProcessingResult
from typing import Dict, Any, List

class MyEmbeddingHandler(EmbeddingHandler):
    @property
    def name(self) -> str:
        return "my_embedding_handler"

    async def process(self, context: ProcessingContext) -> ProcessingResult:
        try:
            # Generate embedding
            embedding = await self.generate_embedding(context.text, context.metadata)

            # Store embedding
            stored = await self.store_embedding(embedding, context)

            return ProcessingResult(
                success=stored,
                stage=self.stage,
                data={"embedding": embedding, "stored": stored}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                stage=self.stage,
                data=None,
                error=str(e)
            )

    def can_handle(self, context: ProcessingContext) -> bool:
        # Only process non-empty text
        return len(context.text.strip()) > 0

    async def generate_embedding(self, text: str, metadata: Dict[str, Any]) -> List[float]:
        # Implement your embedding generation logic
        # Example: use sentence-transformers, OpenAI embeddings, etc.
        return [0.1, 0.2, 0.3, 0.4, 0.5]  # Placeholder

    async def store_embedding(self, embedding: List[float], context: ProcessingContext) -> bool:
        # Implement your storage logic
        # Example: store in vector database, file system, etc.
        return True
```

### Response Generation Handler

Create handlers for AI response generation:

```python
from local_ai.speech_to_text import ResponseGenerationHandler

class MyResponseHandler(ResponseGenerationHandler):
    @property
    def name(self) -> str:
        return "my_response_handler"

    async def process(self, context: ProcessingContext) -> ProcessingResult:
        try:
            response = await self.generate_response(context.text, context)

            return ProcessingResult(
                success=True,
                stage=self.stage,
                data={"response": response}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                stage=self.stage,
                data=None,
                error=str(e)
            )

    def can_handle(self, context: ProcessingContext) -> bool:
        # Only handle questions or help requests
        text_lower = context.text.lower()
        return ("?" in context.text or
                any(word in text_lower for word in ["help", "what", "how"]))

    async def generate_response(self, text: str, context: ProcessingContext) -> str:
        # Implement your response generation logic
        # Example: use Ollama, OpenAI API, local LLM, etc.
        return f"I understand you said: {text}. How can I help?"

    async def get_conversation_context(self, session_id: str) -> List[Dict[str, Any]]:
        # Implement conversation history retrieval
        return []
```

### Text-to-Speech Handler

Create handlers for speech synthesis:

```python
from local_ai.speech_to_text import TextToSpeechHandler

class MyTTSHandler(TextToSpeechHandler):
    @property
    def name(self) -> str:
        return "my_tts_handler"

    async def process(self, context: ProcessingContext) -> ProcessingResult:
        try:
            # Synthesize speech
            audio_data = await self.synthesize_speech(context.text, {})

            # Play audio
            played = await self.play_audio(audio_data)

            return ProcessingResult(
                success=played,
                stage=self.stage,
                data={"audio_generated": True, "played": played}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                stage=self.stage,
                data=None,
                error=str(e)
            )

    def can_handle(self, context: ProcessingContext) -> bool:
        # Can handle any text
        return len(context.text.strip()) > 0

    async def synthesize_speech(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        # Implement your TTS logic
        # Example: use pyttsx3, gTTS, Azure Speech, etc.
        return b"audio_data_placeholder"

    async def play_audio(self, audio_data: bytes) -> bool:
        # Implement audio playback
        # Example: use pygame, pyaudio, etc.
        return True
```

## Registering Handlers

Register your handlers with the speech-to-text service:

```python
from local_ai.speech_to_text import SpeechToTextService

# Create service
service = SpeechToTextService()

# Create and register handlers
embedding_handler = MyEmbeddingHandler()
response_handler = MyResponseHandler()
tts_handler = MyTTSHandler()

service.register_processing_handler(embedding_handler)
service.register_processing_handler(response_handler)
service.register_processing_handler(tts_handler)

# Set up pipeline callback to receive results
def pipeline_callback(results):
    for result in results:
        if result.success:
            print(f"✅ {result.stage.value}: {result.data}")
        else:
            print(f"❌ {result.stage.value}: {result.error}")

service.set_pipeline_callback(pipeline_callback)

# Start listening - transcriptions will automatically trigger the pipeline
await service.start_listening()
```

## Pipeline Processing Flow

When a transcription is completed, the following happens automatically:

1. **Transcription Complete** - Speech-to-text service completes transcription
2. **Context Creation** - ProcessingContext is created with all metadata
3. **Handler Filtering** - Only handlers that can handle the context are selected
4. **Concurrent Processing** - Handlers within each stage run concurrently
5. **Stage Ordering** - Stages are processed in order: EMBEDDING → RESPONSE_GENERATION → TEXT_TO_SPEECH
6. **Result Delivery** - Results are delivered via pipeline callback

## Managing Handlers

```python
# Check registered handlers
handlers = service.get_registered_handlers()
print(handlers)  # {"embedding": ["my_embedding_handler"], ...}

# Unregister a handler
service.unregister_processing_handler("embedding", "my_embedding_handler")

# Get pipeline statistics
stats = service.get_pipeline_stats()
print(f"Processed: {stats['total_processed']}")
print(f"Success rate: {stats['successful_processed'] / stats['total_processed']}")
```

## Error Handling

The pipeline handles errors gracefully:

- **Handler Failures** - Individual handler failures don't stop the pipeline
- **Callback Errors** - Pipeline callback errors are logged but don't crash the service
- **Async Safety** - All processing is async-safe and non-blocking

## Metadata Preservation

The system preserves rich metadata for downstream processing:

```python
# Transcription metadata automatically includes:
{
    "confidence": 0.95,           # Transcription confidence
    "timestamp": 1234567890.0,    # When transcription occurred
    "processing_time": 0.5,       # Time taken for transcription
    "audio_duration": 2.0,        # Duration of audio in seconds
    "sample_rate": 16000,         # Audio sample rate
    "chunk_count": 5,             # Number of audio chunks processed
    "session_id": "session-123",  # Session identifier
    "user_id": "user-456",        # User identifier (if available)
    "additional_metadata": {      # Custom metadata
        "audio_size_bytes": 32000,
        "optimization_target": "balanced",
        "monitoring_enabled": True
    }
}
```

## Example: Complete AI Assistant

Here's a complete example showing how to build an AI assistant with all stages:

```python
import asyncio
from local_ai.speech_to_text import SpeechToTextService

async def main():
    # Create service
    service = SpeechToTextService()

    # Register handlers (using the examples above)
    service.register_processing_handler(MyEmbeddingHandler())
    service.register_processing_handler(MyResponseHandler())
    service.register_processing_handler(MyTTSHandler())

    # Set up callbacks
    def transcription_callback(text):
        print(f"🎤 User: {text}")

    def pipeline_callback(results):
        for result in results:
            if result.stage.value == "response_generation" and result.success:
                print(f"🤖 Assistant: {result.data['response']}")

    service.set_transcription_callback(transcription_callback)
    service.set_pipeline_callback(pipeline_callback)

    # Start the AI assistant
    print("🚀 AI Assistant started. Speak into your microphone...")
    await service.start_listening()

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Stopping AI Assistant...")
        await service.stop_listening()

if __name__ == "__main__":
    asyncio.run(main())
```

This creates a complete voice-activated AI assistant that:

1. Listens to speech and transcribes it
2. Generates semantic embeddings for context
3. Generates AI responses to questions
4. Converts responses back to speech
5. Maintains conversation history and context

The extensible interface system makes it easy to swap out components, add new capabilities, and integrate with different AI services and models.
