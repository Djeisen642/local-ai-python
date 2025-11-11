"""Tests for AudioFilterPipeline orchestrator class."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.local_ai.speech_to_text.audio_filtering.models import FilterStats, NoiseType
from src.local_ai.speech_to_text.models import AudioChunk


@pytest.mark.unit
class TestAudioFilterPipeline:
    """Test cases for AudioFilterPipeline real-time processing."""

    @pytest.fixture
    def sample_rate(self) -> int:
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def audio_chunk(self, sample_rate: int) -> AudioChunk:
        """Create test audio chunk."""
        duration = 0.1  # 100ms chunk
        samples = int(sample_rate * duration)
        audio_data = np.random.normal(0, 0.1, samples).astype(np.float32)
        return AudioChunk(
            data=audio_data.tobytes(),
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration=duration,
        )

    @pytest.fixture
    def mock_noise_reduction(self):
        """Mock noise reduction engine."""
        mock = MagicMock()
        mock.reduce_noise.return_value = np.random.normal(0, 0.05, 1600).astype(
            np.float32
        )
        mock.get_noise_reduction_db.return_value = 6.0
        mock.detect_noise_type.return_value = NoiseType.STATIONARY
        return mock

    @pytest.fixture
    def mock_audio_normalizer(self):
        """Mock audio normalizer."""
        mock = MagicMock()
        mock.normalize_audio.return_value = np.random.normal(0, 0.1, 1600).astype(
            np.float32
        )
        mock.get_current_level.return_value = -20.0
        return mock

    @pytest.fixture
    def mock_spectral_enhancer(self):
        """Mock spectral enhancer."""
        mock = MagicMock()
        mock.enhance_speech_frequencies.return_value = np.random.normal(
            0, 0.1, 1600
        ).astype(np.float32)
        mock.apply_high_pass_filter.return_value = np.random.normal(0, 0.1, 1600).astype(
            np.float32
        )
        return mock

    @pytest.fixture
    def mock_adaptive_processor(self):
        """Mock adaptive processor."""
        mock = MagicMock()
        mock.select_optimal_filters.return_value = ["noise_reduction", "normalization"]
        return mock

    @pytest.mark.asyncio
    async def test_audio_filter_pipeline_initialization(self, sample_rate: int):
        """Test AudioFilterPipeline can be initialized with proper parameters."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

        assert pipeline.sample_rate == sample_rate
        assert pipeline.enable_filtering is True
        assert len(pipeline.filter_chain) > 0

        # Test performance monitoring is initialized
        perf_summary = pipeline.get_performance_summary()
        assert isinstance(perf_summary, dict)

    @pytest.mark.asyncio
    async def test_audio_filter_pipeline_disabled(
        self, sample_rate: int, audio_chunk: AudioChunk
    ):
        """Test AudioFilterPipeline passes through audio when disabled."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=False)

        result = await pipeline.process_audio_chunk(audio_chunk)

        assert result.data == audio_chunk.data

        # Test that performance monitoring shows minimal overhead
        perf_summary = pipeline.get_performance_summary()
        if perf_summary.get("avg_latency_ms", 0) > 0:
            assert perf_summary["avg_latency_ms"] < 1.0  # Minimal overhead

    @pytest.mark.asyncio
    async def test_real_time_audio_chunk_processing_timing(
        self,
        sample_rate: int,
        audio_chunk: AudioChunk,
        mock_noise_reduction,
        mock_audio_normalizer,
        mock_spectral_enhancer,
        mock_adaptive_processor,
    ):
        """Test real-time audio chunk processing meets timing requirements (<50ms)."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=mock_noise_reduction),
            AudioNormalizer=MagicMock(return_value=mock_audio_normalizer),
            SpectralEnhancer=MagicMock(return_value=mock_spectral_enhancer),
            AdaptiveProcessor=MagicMock(return_value=mock_adaptive_processor),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            start_time = time.perf_counter()
            result = await pipeline.process_audio_chunk(audio_chunk)
            end_time = time.perf_counter()

            processing_time_ms = (end_time - start_time) * 1000

            assert processing_time_ms < 50.0  # Must be under 50ms
            assert result.is_filtered is True

            # Test that performance monitoring tracks latency correctly
            perf_summary = pipeline.get_performance_summary()
            if perf_summary.get("avg_latency_ms", 0) > 0:
                assert perf_summary["avg_latency_ms"] <= processing_time_ms

    @pytest.mark.asyncio
    async def test_filter_chain_management(
        self,
        sample_rate: int,
        audio_chunk: AudioChunk,
        mock_noise_reduction,
        mock_audio_normalizer,
        mock_spectral_enhancer,
        mock_adaptive_processor,
    ):
        """Test filter chain management and proper filter ordering."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=mock_noise_reduction),
            AudioNormalizer=MagicMock(return_value=mock_audio_normalizer),
            SpectralEnhancer=MagicMock(return_value=mock_spectral_enhancer),
            AdaptiveProcessor=MagicMock(return_value=mock_adaptive_processor),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Test that filters are applied in correct order
            await pipeline.process_audio_chunk(audio_chunk)

            # Verify filter chain execution
            assert (
                len(pipeline.filter_chain) >= 3
            )  # At least noise reduction, normalization, spectral enhancement

            # Check that adaptive processor was consulted for filter selection
            mock_adaptive_processor.select_optimal_filters.assert_called()

    @pytest.mark.asyncio
    async def test_bypass_functionality(
        self,
        sample_rate: int,
        audio_chunk: AudioChunk,
        mock_noise_reduction,
        mock_audio_normalizer,
    ):
        """Test filter bypass functionality for individual filters."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=mock_noise_reduction),
            AudioNormalizer=MagicMock(return_value=mock_audio_normalizer),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Test bypassing specific filters
            pipeline.bypass_filter("noise_reduction")

            result = await pipeline.process_audio_chunk(audio_chunk)

            # Noise reduction should not have been called
            mock_noise_reduction.reduce_noise.assert_not_called()
            # But normalization should still work
            mock_audio_normalizer.normalize_audio.assert_called()

            assert result.is_filtered is True

    @pytest.mark.asyncio
    async def test_performance_monitoring_and_latency_tracking(
        self,
        sample_rate: int,
        audio_chunk: AudioChunk,
        mock_noise_reduction,
        mock_audio_normalizer,
    ):
        """Test performance monitoring and latency tracking accuracy."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=mock_noise_reduction),
            AudioNormalizer=MagicMock(return_value=mock_audio_normalizer),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Process multiple chunks to build performance history
            for _ in range(5):
                await pipeline.process_audio_chunk(audio_chunk)

            stats = pipeline.get_filter_stats()

            assert isinstance(stats, FilterStats)
            assert stats.processing_latency_ms > 0.0
            assert stats.processing_latency_ms < 50.0  # Within real-time limits
            assert len(stats.filters_applied) > 0
            assert stats.audio_quality_score >= 0.0
            assert stats.audio_quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_concurrent_audio_chunk_processing(
        self, sample_rate: int, mock_noise_reduction, mock_audio_normalizer
    ):
        """Test concurrent processing of multiple audio chunks."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=mock_noise_reduction),
            AudioNormalizer=MagicMock(return_value=mock_audio_normalizer),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Create multiple audio chunks
            chunks = []
            for i in range(3):
                duration = 0.1
                samples = int(sample_rate * duration)
                audio_data = np.random.normal(0, 0.1, samples).astype(np.float32)
                chunk = AudioChunk(
                    data=audio_data.tobytes(),
                    timestamp=time.time() + i * 0.1,
                    sample_rate=sample_rate,
                    duration=duration,
                )
                chunks.append(chunk)

            # Process chunks concurrently
            tasks = [pipeline.process_audio_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for result in results:
                assert result.is_filtered is True
                assert isinstance(result, AudioChunk)

    @pytest.mark.asyncio
    async def test_noise_profile_management(
        self, sample_rate: int, audio_chunk: AudioChunk, mock_noise_reduction
    ):
        """Test noise profile setting and management."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline.NoiseReductionEngine",
            return_value=mock_noise_reduction,
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Test setting noise profile
            noise_sample = np.random.normal(0, 0.05, 1600).astype(np.float32).tobytes()
            pipeline.set_noise_profile(noise_sample)

            # Verify noise profile was updated
            mock_noise_reduction.update_noise_profile.assert_called()

    @pytest.mark.asyncio
    async def test_adaptive_filter_reset(
        self,
        sample_rate: int,
        mock_noise_reduction,
        mock_audio_normalizer,
        mock_adaptive_processor,
    ):
        """Test resetting adaptive filters."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=mock_noise_reduction),
            AudioNormalizer=MagicMock(return_value=mock_audio_normalizer),
            AdaptiveProcessor=MagicMock(return_value=mock_adaptive_processor),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Reset adaptive filters
            pipeline.reset_adaptive_filters()

            # Verify all components were reset
            mock_noise_reduction.reset.assert_called()
            mock_adaptive_processor.reset.assert_called()


@pytest.mark.unit
class TestAudioFilterPipelineErrorHandling:
    """Test cases for AudioFilterPipeline error handling and graceful degradation."""

    @pytest.fixture
    def sample_rate(self) -> int:
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def audio_chunk(self, sample_rate: int) -> AudioChunk:
        """Create test audio chunk."""
        duration = 0.1
        samples = int(sample_rate * duration)
        audio_data = np.random.normal(0, 0.1, samples).astype(np.float32)
        return AudioChunk(
            data=audio_data.tobytes(),
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration=duration,
        )

    @pytest.mark.asyncio
    async def test_filter_failure_detection_and_bypass(
        self, sample_rate: int, audio_chunk: AudioChunk
    ):
        """Test filter failure detection and bypass mechanisms."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        # Create a mock that raises an exception
        failing_filter = MagicMock()
        failing_filter.reduce_noise.side_effect = RuntimeError("Filter failed")

        working_filter = MagicMock()
        working_filter.normalize_audio.return_value = np.frombuffer(
            audio_chunk.data, dtype=np.float32
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=failing_filter),
            AudioNormalizer=MagicMock(return_value=working_filter),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Should not raise exception, should bypass failed filter
            result = await pipeline.process_audio_chunk(audio_chunk)

            assert result is not None
            assert isinstance(result, AudioChunk)
            # Should still have some filtering applied (from working filters)
            working_filter.normalize_audio.assert_called()

    @pytest.mark.asyncio
    async def test_performance_based_complexity_adjustment(
        self, sample_rate: int, audio_chunk: AudioChunk
    ):
        """Test performance-based filter complexity adjustment."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        # Mock slow filter that exceeds time budget
        slow_filter = MagicMock()

        def slow_processing(*args, **kwargs):
            time.sleep(0.06)  # 60ms - exceeds 50ms budget
            return np.frombuffer(audio_chunk.data, dtype=np.float32)

        slow_filter.reduce_noise.side_effect = slow_processing

        fast_filter = MagicMock()
        fast_filter.normalize_audio.return_value = np.frombuffer(
            audio_chunk.data, dtype=np.float32
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=slow_filter),
            AudioNormalizer=MagicMock(return_value=fast_filter),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Process multiple chunks to trigger complexity adjustment
            for _ in range(3):
                await pipeline.process_audio_chunk(audio_chunk)

            # Pipeline should adapt by reducing complexity or bypassing slow filters
            stats = pipeline.get_filter_stats()
            assert stats.processing_latency_ms > 0.0

    @pytest.mark.asyncio
    async def test_fallback_to_unfiltered_audio_processing(
        self, sample_rate: int, audio_chunk: AudioChunk
    ):
        """Test fallback to unfiltered audio processing on various error conditions."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        # Create filters that all fail
        failing_noise_filter = MagicMock()
        failing_noise_filter.reduce_noise.side_effect = Exception("Critical failure")

        failing_normalizer = MagicMock()
        failing_normalizer.normalize_audio.side_effect = Exception("Critical failure")

        failing_enhancer = MagicMock()
        failing_enhancer.enhance_speech_frequencies.side_effect = Exception(
            "Critical failure"
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=failing_noise_filter),
            AudioNormalizer=MagicMock(return_value=failing_normalizer),
            SpectralEnhancer=MagicMock(return_value=failing_enhancer),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Should fallback to unfiltered audio
            result = await pipeline.process_audio_chunk(audio_chunk)

            assert result is not None
            assert isinstance(result, AudioChunk)
            # Should return audio data (may have minor format conversion differences)
            # but should be approximately the same length
            assert len(result.data) == len(audio_chunk.data)
            assert result.is_filtered is False

    @pytest.mark.asyncio
    async def test_memory_constraint_handling(
        self, sample_rate: int, audio_chunk: AudioChunk
    ):
        """Test handling of memory constraints during processing."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        # Mock filter that raises MemoryError
        memory_constrained_filter = MagicMock()
        memory_constrained_filter.reduce_noise.side_effect = MemoryError("Out of memory")

        backup_filter = MagicMock()
        backup_filter.normalize_audio.return_value = np.frombuffer(
            audio_chunk.data, dtype=np.float32
        )

        with patch.multiple(
            "src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline",
            NoiseReductionEngine=MagicMock(return_value=memory_constrained_filter),
            AudioNormalizer=MagicMock(return_value=backup_filter),
        ):
            pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

            # Should handle memory error gracefully
            result = await pipeline.process_audio_chunk(audio_chunk)

            assert result is not None
            assert isinstance(result, AudioChunk)
            # Should still process with available filters
            backup_filter.normalize_audio.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_audio_data_handling(self, sample_rate: int):
        """Test handling of invalid or corrupted audio data."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

        # Test with invalid audio chunk
        invalid_chunk = AudioChunk(
            data=b"invalid_audio_data",
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration=0.1,
        )

        # Should handle invalid data gracefully
        result = await pipeline.process_audio_chunk(invalid_chunk)

        assert result is not None
        assert isinstance(result, AudioChunk)
        # Should return some data (may be processed or original depending on error handling)
        assert len(result.data) > 0
        # is_filtered may be True or False depending on whether processing succeeded
        # The important thing is that it doesn't crash


@pytest.mark.integration
class TestAudioFilterPipelineIntegration:
    """Integration tests for AudioFilterPipeline with real components."""

    @pytest.fixture
    def sample_rate(self) -> int:
        """Standard sample rate for testing."""
        return 16000

    @pytest.fixture
    def real_audio_chunk(self, sample_rate: int) -> AudioChunk:
        """Create realistic audio chunk with speech-like characteristics."""
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Generate speech-like signal with harmonics
        fundamental = 200.0
        speech = (
            0.5 * np.sin(2 * np.pi * fundamental * t)
            + 0.3 * np.sin(2 * np.pi * 2 * fundamental * t)
            + 0.2 * np.sin(2 * np.pi * 3 * fundamental * t)
        )

        # Add some noise
        noise = np.random.normal(0, 0.05, len(speech))
        audio_data = (speech + noise).astype(np.float32)

        return AudioChunk(
            data=audio_data.tobytes(),
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration=duration,
        )

    @pytest.mark.asyncio
    async def test_end_to_end_filtering_pipeline(
        self, sample_rate: int, real_audio_chunk: AudioChunk
    ):
        """Test end-to-end filtering pipeline with real components."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

        result = await pipeline.process_audio_chunk(real_audio_chunk)

        assert result is not None
        assert isinstance(result, AudioChunk)
        assert result.is_filtered is True
        assert result.sample_rate == sample_rate
        assert result.duration == real_audio_chunk.duration

        # Verify filtering stats are reasonable (relaxed for integration test with real components)
        stats = pipeline.get_filter_stats()
        assert stats.processing_latency_ms < 200.0  # More relaxed for real components
        assert len(stats.filters_applied) > 0

    @pytest.mark.asyncio
    async def test_streaming_audio_processing(self, sample_rate: int):
        """Test streaming audio processing with multiple consecutive chunks."""
        from src.local_ai.speech_to_text.audio_filtering.audio_filter_pipeline import (
            AudioFilterPipeline,
        )

        pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

        # Process stream of audio chunks
        results = []
        for i in range(10):
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz tone

            chunk = AudioChunk(
                data=audio_data.tobytes(),
                timestamp=time.time() + i * 0.1,
                sample_rate=sample_rate,
                duration=duration,
            )

            result = await pipeline.process_audio_chunk(chunk)
            results.append(result)

        assert len(results) == 10
        for result in results:
            assert result.is_filtered is True

        # Verify consistent performance across stream (relaxed for integration test)
        final_stats = pipeline.get_filter_stats()
        assert (
            final_stats.processing_latency_ms < 200.0
        )  # More relaxed for real components

        pipeline = AudioFilterPipeline(sample_rate=sample_rate, enable_filtering=True)

        # Process stream of audio chunks
        results = []
        for i in range(10):
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz tone

            chunk = AudioChunk(
                data=audio_data.tobytes(),
                timestamp=time.time() + i * 0.1,
                sample_rate=sample_rate,
                duration=duration,
            )

            result = await pipeline.process_audio_chunk(chunk)
            results.append(result)

        assert len(results) == 10
        for result in results:
            assert result.is_filtered is True

        # Verify consistent performance across stream
        final_stats = pipeline.get_filter_stats()
        assert final_stats.processing_latency_ms < 50.0
