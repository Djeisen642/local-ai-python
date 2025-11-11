#!/usr/bin/env python3
"""Simple test to check filtering effectiveness on noisy audio."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from local_ai.speech_to_text.models import AudioChunk
from local_ai.speech_to_text.service import SpeechToTextService


async def test_noisy_audio_filtering():
    """Test filtering on noisy audio."""

    # Create services
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

    # Initialize services
    service_with_filtering._initialize_components()
    service_without_filtering._initialize_components()

    # Test with noisy audio
    test_files = [
        "synthetic_noise/hello_world/white_noise_5db.wav",
        "synthetic_noise/hello_world/pink_noise_8db.wav",
        "synthetic_noise/hello_world/clicks_heavy.wav",
        "hello_world.wav",  # Clean reference
    ]

    test_data_dir = Path(__file__).parent.parent / "test_data" / "audio"

    for test_file in test_files:
        test_path = test_data_dir / test_file
        if not test_path.exists():
            print(f"Skipping {test_file} - file not found")
            continue

        print(f"\n--- Testing {test_file} ---")

        with open(test_path, "rb") as f:
            audio_data = f.read()

        # Set noise profile for filtering service
        if service_with_filtering._audio_filter_pipeline:
            noise_sample = audio_data[:8000]  # First 0.5 seconds as noise
            service_with_filtering._audio_filter_pipeline.set_noise_profile(noise_sample)

        # Test filtering
        if service_with_filtering._audio_filter_pipeline:
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=16000,
                duration=len(audio_data) / (16000 * 2),
                is_filtered=False,
            )

            filtered_chunk = (
                await service_with_filtering._audio_filter_pipeline.process_audio_chunk(
                    audio_chunk
                )
            )
            print(f"  Filtering applied: {filtered_chunk.is_filtered}")
            print(f"  Audio changed: {audio_data != filtered_chunk.data}")

            # Use filtered audio for transcription
            filtered_audio = filtered_chunk.data
        else:
            filtered_audio = audio_data

        # Transcribe both versions
        try:
            result_filtered = (
                await service_with_filtering._transcriber.transcribe_audio_with_result(
                    filtered_audio
                )
            )
            result_unfiltered = (
                await service_without_filtering._transcriber.transcribe_audio_with_result(
                    audio_data
                )
            )

            print(
                f"  Filtered result: '{result_filtered.text}' (confidence: {result_filtered.confidence:.3f})"
            )
            print(
                f"  Unfiltered result: '{result_unfiltered.text}' (confidence: {result_unfiltered.confidence:.3f})"
            )
            print(
                f"  Results different: {result_filtered.text != result_unfiltered.text}"
            )
            print(
                f"  Confidence improvement: {result_filtered.confidence - result_unfiltered.confidence:+.3f}"
            )

        except Exception as e:
            print(f"  Error during transcription: {e}")


if __name__ == "__main__":
    asyncio.run(test_noisy_audio_filtering())
