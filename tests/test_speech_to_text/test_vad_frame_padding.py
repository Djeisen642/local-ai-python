"""Tests for VAD frame padding functionality in service.py (TDD - Red phase)."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_ai.speech_to_text.config import DEFAULT_SAMPLE_RATE, VAD_FRAME_DURATION
from local_ai.speech_to_text.service import SpeechToTextService


@pytest.mark.unit
class TestVADFramePaddingEnabled:
    """Test cases for VAD frame padding when enabled (Requirement 2.1, 2.2)."""

    @pytest.mark.asyncio
    async def test_incomplete_frame_is_padded_with_zeros(self) -> None:
        """
        Test that incomplete VAD frames are padded with zeros when
        VAD_PAD_INCOMPLETE_FRAMES is True.

        Requirements: 2.1, 2.2
        """
        # Calculate VAD frame size in bytes (16-bit audio = 2 bytes per sample)
        vad_frame_size_samples = int(DEFAULT_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
        vad_frame_size_bytes = vad_frame_size_samples * 2

        # Create an incomplete frame (e.g., 75% of a full frame)
        incomplete_frame_size = int(vad_frame_size_bytes * 0.75)
        incomplete_audio_chunk = b"\x01\x02" * (incomplete_frame_size // 2)

        # Verify the chunk is indeed incomplete
        assert len(incomplete_audio_chunk) < vad_frame_size_bytes
        assert len(incomplete_audio_chunk) > 0

        # Create service with VAD_PAD_INCOMPLETE_FRAMES enabled (default)
        service = SpeechToTextService()

        # Mock components
        with patch("local_ai.speech_to_text.config.VAD_PAD_INCOMPLETE_FRAMES", True):
            service._audio_capture = Mock()
            service._audio_capture.get_audio_chunk = AsyncMock(
                side_effect=[incomplete_audio_chunk, None]
            )
            service._audio_capture.is_capturing = Mock(return_value=True)

            service._vad = Mock()
            service._vad.frame_size = vad_frame_size_samples
            service._vad.is_speech = Mock(return_value=True)

            service._transcriber = Mock()
            service._transcriber.transcribe_audio_with_result = AsyncMock()

            service._listening = True

            # Track VAD calls to verify padded frame is processed
            vad_calls = []

            def track_vad_call(frame):
                vad_calls.append(frame)
                return True

            service._vad.is_speech = Mock(side_effect=track_vad_call)

            # Run the audio processing pipeline briefly
            processing_task = asyncio.create_task(service._process_audio_pipeline())

            # Let it process for a short time
            await asyncio.sleep(0.1)

            # Stop the service
            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Verify that VAD was called with a padded frame
            # The padded frame should be exactly vad_frame_size_bytes
            assert len(vad_calls) > 0, "VAD should have been called with padded frame"

            # Check that at least one call had the correct size (padded)
            padded_frames = [f for f in vad_calls if len(f) == vad_frame_size_bytes]
            assert len(padded_frames) > 0, (
                "At least one frame should be padded to correct size"
            )

            # Verify the padded portion contains zeros
            padded_frame = padded_frames[0]
            # The original data should be at the start
            assert padded_frame[:incomplete_frame_size] == incomplete_audio_chunk
            # The padding should be zeros
            padding = padded_frame[incomplete_frame_size:]
            assert padding == b"\x00" * len(padding), "Padding should be zeros"

    @pytest.mark.asyncio
    async def test_padded_frame_is_processed_by_vad(self) -> None:
        """
        Test that padded frames are actually processed by VAD.

        Requirements: 2.1, 2.2
        """
        # Calculate VAD frame size
        vad_frame_size_samples = int(DEFAULT_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
        vad_frame_size_bytes = vad_frame_size_samples * 2

        # Create an incomplete frame
        incomplete_frame_size = int(vad_frame_size_bytes * 0.6)
        incomplete_audio_chunk = b"\x03\x04" * (incomplete_frame_size // 2)

        service = SpeechToTextService()

        with patch("local_ai.speech_to_text.config.VAD_PAD_INCOMPLETE_FRAMES", True):
            service._audio_capture = Mock()
            service._audio_capture.get_audio_chunk = AsyncMock(
                side_effect=[incomplete_audio_chunk, None]
            )

            service._vad = Mock()
            service._vad.frame_size = vad_frame_size_samples
            vad_call_count = 0

            def count_vad_calls(frame):
                nonlocal vad_call_count
                vad_call_count += 1
                return True

            service._vad.is_speech = Mock(side_effect=count_vad_calls)

            service._transcriber = Mock()
            service._transcriber.transcribe_audio_with_result = AsyncMock()

            service._listening = True

            # Run the pipeline
            processing_task = asyncio.create_task(service._process_audio_pipeline())
            await asyncio.sleep(0.1)

            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Verify VAD was called (meaning the padded frame was processed)
            assert vad_call_count > 0, "VAD should be called with padded incomplete frame"

    @pytest.mark.asyncio
    async def test_multiple_incomplete_frames_are_all_padded(self) -> None:
        """
        Test that multiple incomplete frames in sequence are all padded.

        Requirements: 2.1, 2.2
        """
        vad_frame_size_samples = int(DEFAULT_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
        vad_frame_size_bytes = vad_frame_size_samples * 2

        # Create multiple incomplete frames of different sizes
        incomplete_frame_1 = b"\x01\x02" * (int(vad_frame_size_bytes * 0.5) // 2)
        incomplete_frame_2 = b"\x03\x04" * (int(vad_frame_size_bytes * 0.7) // 2)
        incomplete_frame_3 = b"\x05\x06" * (int(vad_frame_size_bytes * 0.3) // 2)

        service = SpeechToTextService()

        with patch("local_ai.speech_to_text.config.VAD_PAD_INCOMPLETE_FRAMES", True):
            service._audio_capture = Mock()
            service._audio_capture.get_audio_chunk = AsyncMock(
                side_effect=[
                    incomplete_frame_1,
                    incomplete_frame_2,
                    incomplete_frame_3,
                    None,
                ]
            )

            service._vad = Mock()
            service._vad.frame_size = vad_frame_size_samples

            vad_calls = []

            def track_vad_calls(frame):
                vad_calls.append(frame)
                return True

            service._vad.is_speech = Mock(side_effect=track_vad_calls)

            service._transcriber = Mock()
            service._transcriber.transcribe_audio_with_result = AsyncMock()

            service._listening = True

            processing_task = asyncio.create_task(service._process_audio_pipeline())
            await asyncio.sleep(0.2)

            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # All incomplete frames should be padded to correct size
            padded_frames = [f for f in vad_calls if len(f) == vad_frame_size_bytes]
            assert len(padded_frames) >= 3, "All incomplete frames should be padded"


@pytest.mark.unit
class TestVADFramePaddingDisabled:
    """Test cases for VAD frame padding when disabled (Requirement 2.5)."""

    @pytest.mark.asyncio
    async def test_incomplete_frame_is_skipped_when_padding_disabled(self) -> None:
        """
        Test that incomplete VAD frames are skipped when
        VAD_PAD_INCOMPLETE_FRAMES is False.

        Requirements: 2.5
        """
        vad_frame_size_samples = int(DEFAULT_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
        vad_frame_size_bytes = vad_frame_size_samples * 2

        # Create an incomplete frame
        incomplete_frame_size = int(vad_frame_size_bytes * 0.75)
        incomplete_audio_chunk = b"\x01\x02" * (incomplete_frame_size // 2)

        # Create a complete frame for comparison
        complete_audio_chunk = b"\x03\x04" * (vad_frame_size_bytes // 2)

        service = SpeechToTextService()

        with patch("local_ai.speech_to_text.config.VAD_PAD_INCOMPLETE_FRAMES", False):
            service._audio_capture = Mock()
            service._audio_capture.get_audio_chunk = AsyncMock(
                side_effect=[incomplete_audio_chunk, complete_audio_chunk, None]
            )

            service._vad = Mock()
            service._vad.frame_size = vad_frame_size_samples

            vad_calls = []

            def track_vad_calls(frame):
                vad_calls.append(frame)
                return True

            service._vad.is_speech = Mock(side_effect=track_vad_calls)

            service._transcriber = Mock()
            service._transcriber.transcribe_audio_with_result = AsyncMock()

            service._listening = True

            processing_task = asyncio.create_task(service._process_audio_pipeline())
            await asyncio.sleep(0.15)

            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Verify that only complete frames were processed
            # Incomplete frames should be skipped
            assert len(vad_calls) > 0, "VAD should be called with complete frames"

            # Check that no incomplete frames were processed
            incomplete_frames = [f for f in vad_calls if len(f) < vad_frame_size_bytes]
            assert len(incomplete_frames) == 0, (
                "Incomplete frames should be skipped when padding is disabled"
            )

            # Verify complete frame was processed
            complete_frames = [f for f in vad_calls if len(f) == vad_frame_size_bytes]
            assert len(complete_frames) > 0, "Complete frames should still be processed"

    @pytest.mark.asyncio
    async def test_no_padding_occurs_when_disabled(self) -> None:
        """
        Test that no zero-padding occurs when VAD_PAD_INCOMPLETE_FRAMES is False.

        Requirements: 2.5
        """
        vad_frame_size_samples = int(DEFAULT_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
        vad_frame_size_bytes = vad_frame_size_samples * 2

        incomplete_frame_size = int(vad_frame_size_bytes * 0.5)
        incomplete_audio_chunk = b"\xff\xfe" * (incomplete_frame_size // 2)

        service = SpeechToTextService()

        with patch("local_ai.speech_to_text.config.VAD_PAD_INCOMPLETE_FRAMES", False):
            service._audio_capture = Mock()
            service._audio_capture.get_audio_chunk = AsyncMock(
                side_effect=[incomplete_audio_chunk, None]
            )

            service._vad = Mock()
            service._vad.frame_size = vad_frame_size_samples

            vad_calls = []

            def track_vad_calls(frame):
                vad_calls.append(frame)
                return True

            service._vad.is_speech = Mock(side_effect=track_vad_calls)

            service._transcriber = Mock()
            service._transcriber.transcribe_audio_with_result = AsyncMock()

            service._listening = True

            processing_task = asyncio.create_task(service._process_audio_pipeline())
            await asyncio.sleep(0.1)

            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Verify no frames with zero padding were processed
            for frame in vad_calls:
                # If a frame was processed, it should not contain the padding pattern
                # (original data followed by zeros)
                if len(frame) == vad_frame_size_bytes:
                    # This would be a padded frame - should not exist
                    padding_start = len(incomplete_audio_chunk)
                    if padding_start < len(frame):
                        padding = frame[padding_start:]
                        # Padding should NOT be all zeros (frame should be skipped instead)
                        assert padding != b"\x00" * len(padding), (
                            "No zero-padding should occur when disabled"
                        )


@pytest.mark.unit
class TestSampleRatePropagation:
    """Test cases for sample rate propagation through pipeline (Requirement 3.1, 3.2, 3.3)."""

    @pytest.mark.asyncio
    async def test_service_passes_sample_rate_to_transcriber(self) -> None:
        """
        Test that Service passes sample rate to Transcriber during transcription.

        Requirements: 3.1, 3.2, 3.3
        """
        service = SpeechToTextService()

        # Mock components
        service._audio_capture = Mock()
        service._audio_capture.get_audio_chunk = AsyncMock(
            side_effect=[b"\x01\x02" * 8000, None]  # Enough for one VAD frame
        )

        service._vad = Mock()
        service._vad.frame_size = 480  # 30ms at 16kHz
        service._vad.is_speech = Mock(return_value=True)

        service._transcriber = Mock()
        transcribe_calls = []

        async def track_transcribe_call(*args, **kwargs):
            transcribe_calls.append({"args": args, "kwargs": kwargs})
            from local_ai.speech_to_text.models import TranscriptionResult

            return TranscriptionResult(
                text="test", confidence=0.9, timestamp=0.0, processing_time=0.1
            )

        service._transcriber.transcribe_audio_with_result = AsyncMock(
            side_effect=track_transcribe_call
        )

        service._listening = True

        # Mock optimizer to return known sample rate
        with patch("local_ai.speech_to_text.service.get_optimizer") as mock_optimizer:
            mock_opt_instance = Mock()
            mock_opt_instance.get_optimized_pipeline_config.return_value = {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "vad_frame_duration": VAD_FRAME_DURATION,
                "max_silence_duration": 2.0,
                "max_audio_buffer_size": 10,
                "min_speech_duration": 0.5,
                "processing_interval": 0.01,
            }
            mock_optimizer.return_value = mock_opt_instance

            processing_task = asyncio.create_task(service._process_audio_pipeline())
            await asyncio.sleep(0.2)

            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Verify transcriber was called with source_sample_rate parameter
            assert len(transcribe_calls) > 0, (
                "Transcriber should be called with sample rate"
            )

            # Check that source_sample_rate was passed
            for call in transcribe_calls:
                assert "source_sample_rate" in call["kwargs"], (
                    "source_sample_rate should be passed to transcriber"
                )
                assert call["kwargs"]["source_sample_rate"] == DEFAULT_SAMPLE_RATE, (
                    f"Sample rate should be {DEFAULT_SAMPLE_RATE}"
                )

    @pytest.mark.asyncio
    async def test_sample_rate_consistency_throughout_pipeline(self) -> None:
        """
        Test that sample rate remains consistent throughout the pipeline.

        Requirements: 3.1, 3.2, 3.3
        """
        expected_sample_rate = DEFAULT_SAMPLE_RATE

        service = SpeechToTextService()

        # Mock components with sample rate tracking
        service._audio_capture = Mock()
        service._audio_capture.get_audio_chunk = AsyncMock(
            side_effect=[b"\x01\x02" * 8000, None]
        )

        service._vad = Mock()
        service._vad.frame_size = 480
        service._vad.is_speech = Mock(return_value=True)

        service._transcriber = Mock()
        sample_rates_seen = []

        async def track_sample_rate(*args, **kwargs):
            if "source_sample_rate" in kwargs:
                sample_rates_seen.append(kwargs["source_sample_rate"])
            from local_ai.speech_to_text.models import TranscriptionResult

            return TranscriptionResult(
                text="test", confidence=0.9, timestamp=0.0, processing_time=0.1
            )

        service._transcriber.transcribe_audio_with_result = AsyncMock(
            side_effect=track_sample_rate
        )

        service._listening = True

        with patch("local_ai.speech_to_text.service.get_optimizer") as mock_optimizer:
            mock_opt_instance = Mock()
            mock_opt_instance.get_optimized_pipeline_config.return_value = {
                "sample_rate": expected_sample_rate,
                "vad_frame_duration": VAD_FRAME_DURATION,
                "max_silence_duration": 2.0,
                "max_audio_buffer_size": 10,
                "min_speech_duration": 0.5,
                "processing_interval": 0.01,
            }
            mock_optimizer.return_value = mock_opt_instance

            processing_task = asyncio.create_task(service._process_audio_pipeline())
            await asyncio.sleep(0.2)

            service._listening = False
            await asyncio.sleep(0.05)
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Verify all sample rates are consistent
            assert len(sample_rates_seen) > 0, "Sample rate should be tracked"
            for rate in sample_rates_seen:
                assert rate == expected_sample_rate, (
                    f"Sample rate should be consistent: {rate} != {expected_sample_rate}"
                )
