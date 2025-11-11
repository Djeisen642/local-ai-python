"""Simple test to debug audio filtering functionality."""

import asyncio
import time
from pathlib import Path

import pytest

from src.local_ai.speech_to_text.models import AudioChunk
from src.local_ai.speech_to_text.service import SpeechToTextService


@pytest.mark.asyncio
async def test_audio_filtering_debug():
    """Debug test to check if audio filtering is actually working."""

    # Create services with and without filtering
    service_with_filtering = SpeechToTextService(
        optimization_target="accuracy",
        enable_monitoring=True,
        use_cache=True,
        force_cpu=True,
        enable_filtering=True,
    )

    service_without_filtering = SpeechToTextService(
        optimization_target="accuracy",
        enable_monitoring=True,
        use_cache=True,
        force_cpu=True,
        enable_filtering=False,
    )

    # Initialize both services
    service_with_filtering._initialize_components()
    service_without_filtering._initialize_components()

    print(
        f"\nService with filtering - filtering enabled: {service_with_filtering.is_filtering_enabled()}"
    )
    print(
        f"Service without filtering - filtering enabled: {service_without_filtering.is_filtering_enabled()}"
    )

    # Check if filter pipeline exists
    print(
        f"Filter pipeline exists (with): {service_with_filtering._audio_filter_pipeline is not None}"
    )
    print(
        f"Filter pipeline exists (without): {service_without_filtering._audio_filter_pipeline is not None}"
    )

    # Load a noisy test audio file
    test_audio_path = (
        Path(__file__).parent.parent
        / "test_data"
        / "audio"
        / "synthetic_noise"
        / "hello_world"
        / "white_noise_5db.wav"
    )
    if not test_audio_path.exists():
        # Fallback to clean audio
        test_audio_path = (
            Path(__file__).parent.parent / "test_data" / "audio" / "hello_world.wav"
        )
        if not test_audio_path.exists():
            pytest.skip(f"Test audio file not found: {test_audio_path}")

    with open(test_audio_path, "rb") as f:
        audio_data = f.read()

    print(f"Using test file: {test_audio_path.name}")

    print(f"Original audio size: {len(audio_data)} bytes")

    # Test filtering directly if available
    if service_with_filtering._audio_filter_pipeline:
        try:
            # Create AudioChunk
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=16000,
                duration=len(audio_data) / (16000 * 2),
                is_filtered=False,
            )

            print(
                f"Input chunk - filtered: {audio_chunk.is_filtered}, size: {len(audio_chunk.data)}"
            )

            # Process through filter pipeline
            filtered_chunk = (
                await service_with_filtering._audio_filter_pipeline.process_audio_chunk(
                    audio_chunk
                )
            )

            print(
                f"Output chunk - filtered: {filtered_chunk.is_filtered}, size: {len(filtered_chunk.data)}"
            )
            print(f"Audio data changed: {audio_chunk.data != filtered_chunk.data}")

            # Get filter stats
            filter_stats = (
                service_with_filtering._audio_filter_pipeline.get_filter_stats()
            )
            print(f"Filter stats: {filter_stats}")

        except Exception as e:
            print(f"Error testing filter pipeline: {e}")
            import traceback

            traceback.print_exc()

    # Test transcription with both services
    print("\nTesting transcription...")

    try:
        result_with_filtering = (
            await service_with_filtering._transcriber.transcribe_audio_with_result(
                audio_data
            )
        )
        result_without_filtering = (
            await service_without_filtering._transcriber.transcribe_audio_with_result(
                audio_data
            )
        )

        print(
            f"With filtering: '{result_with_filtering.text}' (confidence: {result_with_filtering.confidence:.3f})"
        )
        print(
            f"Without filtering: '{result_without_filtering.text}' (confidence: {result_without_filtering.confidence:.3f})"
        )
        print(
            f"Results identical: {result_with_filtering.text == result_without_filtering.text}"
        )

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_audio_filtering_debug())
