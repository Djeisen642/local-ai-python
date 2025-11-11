"""AudioFilterPipeline orchestrator for real-time audio processing."""

import asyncio
import time
from typing import Dict, List, Optional

import numpy as np

from ..config import (
    AUDIO_FILTER_AGGRESSIVENESS,
    AUDIO_FILTER_MAX_LATENCY_MS,
    FILTER_PROCESSING_TIMEOUT_SEC,
)
from ..models import AudioChunk
from .adaptive_processor import AdaptiveProcessor
from .audio_normalizer import AudioNormalizer
from .interfaces import AudioFilterInterface
from .models import FilterStats, NoiseType
from .noise_reduction import NoiseReductionEngine
from .performance_monitor import AudioFilterPerformanceMonitor
from .spectral_enhancer import SpectralEnhancer


class AudioFilterPipeline(AudioFilterInterface):
    """
    Main orchestrator for real-time audio filtering pipeline.

    Integrates noise reduction, normalization, spectral enhancement, and adaptive
    processing to provide comprehensive audio filtering with performance monitoring
    and graceful degradation capabilities.
    """

    def __init__(
        self,
        sample_rate: int,
        enable_filtering: bool = True,
        max_latency_ms: float = AUDIO_FILTER_MAX_LATENCY_MS,
        aggressiveness: float = AUDIO_FILTER_AGGRESSIVENESS,
    ) -> None:
        """
        Initialize the audio filter pipeline.

        Args:
            sample_rate: Audio sample rate in Hz
            enable_filtering: Whether to enable audio filtering
            max_latency_ms: Maximum allowed processing latency in milliseconds
            aggressiveness: Overall filtering aggressiveness (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.enable_filtering = enable_filtering
        self.max_latency_ms = max_latency_ms
        self.aggressiveness = aggressiveness

        # Performance monitoring and optimization
        self.performance_monitor = AudioFilterPerformanceMonitor(
            target_latency_ms=max_latency_ms
        )

        # Filter components - will be initialized with adaptive parameters
        self._noise_reduction: Optional[NoiseReductionEngine] = None
        self._audio_normalizer: Optional[AudioNormalizer] = None
        self._spectral_enhancer: Optional[SpectralEnhancer] = None
        self._adaptive_processor = AdaptiveProcessor()

        # Initialize components with current performance settings
        self._initialize_filter_components()

        # Filter chain management
        self.filter_chain: List[str] = []
        self._bypassed_filters: set[str] = set()
        self._filter_stats: Dict[str, Dict] = {}

        # Initialize filter chain
        self._initialize_filter_chain()

        # Error handling and graceful degradation
        self._filter_failures: Dict[str, int] = {}
        self._max_failures_per_filter = 3

        # Processing lock for thread safety
        self._processing_lock = asyncio.Lock()

        # Performance monitoring will be started when first used
        self._monitoring_started = False

    def _initialize_filter_components(self) -> None:
        """Initialize filter components with adaptive parameters."""
        # Get current performance recommendations
        complexity_level = self.performance_monitor.get_current_complexity_level()
        aggressiveness = self.performance_monitor.get_recommended_aggressiveness()

        # Initialize components with performance-optimized settings
        self._noise_reduction = NoiseReductionEngine(
            sample_rate=self.sample_rate, aggressiveness=aggressiveness
        )
        self._audio_normalizer = AudioNormalizer(
            target_level=-20.0, max_gain=20.0, sample_rate=self.sample_rate
        )
        self._spectral_enhancer = SpectralEnhancer(sample_rate=self.sample_rate)

    def _initialize_filter_chain(self) -> None:
        """Initialize the filter chain based on performance recommendations."""
        if not self.enable_filtering:
            return

        # Get recommended filters from performance monitor
        recommended_filters = self.performance_monitor.get_recommended_filters()

        # Map performance monitor filter names to pipeline filter names
        filter_mapping = {
            "normalization": "normalization",
            "high_pass_filter": "spectral_enhancement",
            "light_noise_reduction": "noise_reduction",
            "noise_reduction": "noise_reduction",
            "aggressive_noise_reduction": "noise_reduction",
            "spectral_enhancement": "spectral_enhancement",
            "transient_suppression": "spectral_enhancement",
        }

        # Build filter chain from recommendations
        self.filter_chain = []
        for filter_name in recommended_filters:
            mapped_name = filter_mapping.get(filter_name, filter_name)
            if mapped_name not in self.filter_chain:
                self.filter_chain.append(mapped_name)

        # Ensure we have at least normalization if filtering is enabled
        if not self.filter_chain and self.enable_filtering:
            self.filter_chain = ["normalization"]

        # Initialize filter stats
        for filter_name in self.filter_chain:
            self._filter_stats[filter_name] = {
                "calls": 0,
                "total_time_ms": 0.0,
                "failures": 0,
                "last_success": True,
            }

    async def process_audio_chunk(self, audio_chunk: AudioChunk) -> AudioChunk:
        """
        Process an audio chunk through the filtering pipeline.

        Args:
            audio_chunk: Input audio chunk to process

        Returns:
            Processed audio chunk with filtering applied
        """
        if not self.enable_filtering:
            return audio_chunk

        # Check if we should bypass filtering due to performance issues
        if self.performance_monitor.should_bypass_filtering():
            return audio_chunk

        # Record processing start for performance monitoring
        start_time = self.performance_monitor.record_processing_start()

        async with self._processing_lock:
            try:
                # Apply processing timeout to prevent blocking
                return await asyncio.wait_for(
                    self._process_audio_chunk_internal(audio_chunk, start_time),
                    timeout=FILTER_PROCESSING_TIMEOUT_SEC,
                )

            except asyncio.TimeoutError:
                # Handle timeout by recording failure and returning original audio
                self.performance_monitor.record_processing_end(
                    start_time, success=False, quality_score=0.0
                )
                return self._handle_critical_failure(
                    audio_chunk, Exception("Processing timeout")
                )
            except Exception as e:
                # Handle other exceptions
                self.performance_monitor.record_processing_end(
                    start_time, success=False, quality_score=0.0
                )
                return self._handle_critical_failure(audio_chunk, e)

    async def _ensure_monitoring_started(self) -> None:
        """Ensure performance monitoring is started."""
        if not self._monitoring_started:
            asyncio.create_task(self.performance_monitor.start_monitoring())
            self._monitoring_started = True

    async def _process_audio_chunk_internal(
        self, audio_chunk: AudioChunk, start_time: float
    ) -> AudioChunk:
        """Internal processing method with performance monitoring."""
        # Ensure monitoring is started
        await self._ensure_monitoring_started()

        # Update filter configuration based on current performance
        await self._update_filter_configuration()

        # Convert audio data to numpy array
        # WAV files are typically 16-bit integers, convert to float32 for processing
        try:
            # Try 16-bit integer first (most common for WAV files)
            audio_int16 = np.frombuffer(audio_chunk.data, dtype=np.int16)
            audio_data = audio_int16.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        except ValueError:
            # Fallback to float32 if data is already in that format
            audio_data = np.frombuffer(audio_chunk.data, dtype=np.float32)

        # Apply adaptive filter selection
        selected_filters = await self._select_filters_for_chunk(audio_data)

        # Process through filter chain
        processed_audio, any_filter_applied = await self._apply_filter_chain(
            audio_data, selected_filters
        )

        # Calculate quality score (simplified)
        quality_score = self._calculate_quality_score(
            audio_data, processed_audio, any_filter_applied
        )

        # Create result chunk
        # Convert back to 16-bit integer format for compatibility
        processed_audio_clipped = np.clip(processed_audio, -1.0, 1.0)
        processed_audio_int16 = (processed_audio_clipped * 32767).astype(np.int16)

        result_chunk = AudioChunk(
            data=processed_audio_int16.tobytes(),
            timestamp=audio_chunk.timestamp,
            sample_rate=audio_chunk.sample_rate,
            duration=audio_chunk.duration,
            noise_level=audio_chunk.noise_level,
            signal_level=audio_chunk.signal_level,
            snr_db=audio_chunk.snr_db,
            is_filtered=any_filter_applied,
        )

        # Record successful processing
        self.performance_monitor.record_processing_end(
            start_time,
            success=True,
            quality_score=quality_score,
            filters_applied=selected_filters,
        )

        return result_chunk

    async def _select_filters_for_chunk(self, audio_data: np.ndarray) -> List[str]:
        """
        Select optimal filters for the current audio chunk.

        Args:
            audio_data: Audio data to analyze

        Returns:
            List of filter names to apply
        """
        try:
            # Analyze audio characteristics
            profile = self._adaptive_processor.analyze_audio_characteristics(audio_data)

            # Get recommended filters
            recommended_filters = self._adaptive_processor.select_optimal_filters(profile)

            # Map adaptive processor filter names to pipeline filter names
            mapped_filters = self._map_adaptive_filters_to_pipeline(recommended_filters)

            # Filter out bypassed filters and failed filters
            selected_filters = []
            for filter_name in mapped_filters:
                if (
                    filter_name in self.filter_chain
                    and filter_name not in self._bypassed_filters
                    and self._filter_failures.get(filter_name, 0)
                    < self._max_failures_per_filter
                ):
                    selected_filters.append(filter_name)

            return selected_filters

        except Exception:
            # Fallback to default filter chain on analysis failure
            return [f for f in self.filter_chain if f not in self._bypassed_filters]

    async def _apply_filter_chain(
        self, audio_data: np.ndarray, selected_filters: List[str]
    ) -> tuple[np.ndarray, bool]:
        """
        Apply the selected filters to audio data.

        Args:
            audio_data: Input audio data
            selected_filters: List of filters to apply

        Returns:
            Tuple of (processed audio data, whether any filters were applied successfully)
        """
        processed_audio = audio_data.copy()
        any_filter_applied = False

        for filter_name in selected_filters:
            try:
                filter_start_time = time.perf_counter()

                # Apply specific filter
                if filter_name == "noise_reduction":
                    processed_audio = self._noise_reduction.reduce_noise(processed_audio)
                elif filter_name == "normalization":
                    processed_audio = self._audio_normalizer.normalize_audio(
                        processed_audio
                    )
                elif filter_name == "spectral_enhancement":
                    processed_audio = self._spectral_enhancer.enhance_speech_frequencies(
                        processed_audio
                    )

                # Update filter stats
                filter_time_ms = (time.perf_counter() - filter_start_time) * 1000
                self._update_filter_stats(filter_name, filter_time_ms, success=True)
                any_filter_applied = True

                # Check if filter is taking too long
                if filter_time_ms > self.max_latency_ms / len(selected_filters):
                    self._handle_slow_filter(filter_name)

            except Exception as e:
                # Handle individual filter failure
                self._handle_filter_failure(filter_name, e)
                # Continue with remaining filters
                continue

        return processed_audio, any_filter_applied

    def _update_filter_stats(
        self, filter_name: str, processing_time_ms: float, success: bool
    ) -> None:
        """Update statistics for a specific filter."""
        if filter_name not in self._filter_stats:
            self._filter_stats[filter_name] = {
                "calls": 0,
                "total_time_ms": 0.0,
                "failures": 0,
                "last_success": True,
            }

        stats = self._filter_stats[filter_name]
        stats["calls"] += 1
        stats["total_time_ms"] += processing_time_ms
        stats["last_success"] = success

        if not success:
            stats["failures"] += 1

    def _handle_filter_failure(self, filter_name: str, error: Exception) -> None:
        """Handle failure of an individual filter."""
        self._filter_failures[filter_name] = self._filter_failures.get(filter_name, 0) + 1
        self._update_filter_stats(filter_name, 0.0, success=False)

        # If filter has failed too many times, bypass it
        if self._filter_failures[filter_name] >= self._max_failures_per_filter:
            self._bypassed_filters.add(filter_name)

    def _handle_slow_filter(self, filter_name: str) -> None:
        """Handle a filter that's processing too slowly."""
        # For now, just track it - could implement complexity reduction here
        pass

    def _handle_critical_failure(
        self, audio_chunk: AudioChunk, error: Exception
    ) -> AudioChunk:
        """Handle critical pipeline failure by returning unfiltered audio."""
        return AudioChunk(
            data=audio_chunk.data,
            timestamp=audio_chunk.timestamp,
            sample_rate=audio_chunk.sample_rate,
            duration=audio_chunk.duration,
            noise_level=audio_chunk.noise_level,
            signal_level=audio_chunk.signal_level,
            snr_db=audio_chunk.snr_db,
            is_filtered=False,
        )

    def _map_adaptive_filters_to_pipeline(self, adaptive_filters: List[str]) -> List[str]:
        """
        Map adaptive processor filter names to pipeline filter names.

        Args:
            adaptive_filters: Filter names from adaptive processor

        Returns:
            Mapped filter names that the pipeline recognizes
        """
        # Mapping from adaptive processor names to pipeline names
        filter_mapping = {
            # Normalization variants
            "light_normalization": "normalization",
            "normalization": "normalization",
            "agc": "normalization",
            "dynamic_range_compression": "normalization",
            # Noise reduction variants
            "noise_reduction": "noise_reduction",
            "light_noise_reduction": "noise_reduction",
            "aggressive_noise_reduction": "noise_reduction",
            "spectral_subtraction": "noise_reduction",
            "wiener_filter": "noise_reduction",
            # Speech enhancement variants
            "speech_enhancement": "spectral_enhancement",
            "spectral_enhancement": "spectral_enhancement",
            # Transient noise variants (map to noise_reduction)
            "transient_suppression": "noise_reduction",
            "click_removal": "noise_reduction",
            "impulse_filter": "noise_reduction",
            # Mechanical noise variants
            "harmonic_filter": "noise_reduction",
            "notch_filter": "noise_reduction",
            # High-pass filter (map to noise_reduction)
            "high_pass_filter": "noise_reduction",
        }

        mapped_filters = []
        for filter_name in adaptive_filters:
            mapped_name = filter_mapping.get(filter_name, filter_name)
            if mapped_name in self.filter_chain and mapped_name not in mapped_filters:
                mapped_filters.append(mapped_name)

        return mapped_filters

    async def _update_filter_configuration(self) -> None:
        """Update filter configuration based on current performance."""
        # Get current performance recommendations
        complexity_level = self.performance_monitor.get_current_complexity_level()
        recommended_aggressiveness = (
            self.performance_monitor.get_recommended_aggressiveness()
        )
        recommended_max_latency = self.performance_monitor.get_recommended_max_latency()

        # Update aggressiveness if it has changed significantly
        if abs(self.aggressiveness - recommended_aggressiveness) > 0.1:
            self.aggressiveness = recommended_aggressiveness

            # Update filter components with new aggressiveness
            if self._noise_reduction:
                self._noise_reduction.aggressiveness = recommended_aggressiveness

        # Update max latency
        self.max_latency_ms = recommended_max_latency

        # Update filter chain if recommendations have changed
        recommended_filters = self.performance_monitor.get_recommended_filters()
        current_mapped_filters = self._get_current_mapped_filters()

        if set(recommended_filters) != set(current_mapped_filters):
            self._update_filter_chain_from_recommendations(recommended_filters)

    def _get_current_mapped_filters(self) -> List[str]:
        """Get current filter chain mapped to performance monitor names."""
        reverse_mapping = {
            "normalization": ["normalization"],
            "noise_reduction": [
                "light_noise_reduction",
                "noise_reduction",
                "aggressive_noise_reduction",
            ],
            "spectral_enhancement": [
                "high_pass_filter",
                "spectral_enhancement",
                "transient_suppression",
            ],
        }

        mapped_filters = []
        for filter_name in self.filter_chain:
            if filter_name in reverse_mapping:
                # Use the first mapping as representative
                mapped_filters.append(reverse_mapping[filter_name][0])

        return mapped_filters

    def _update_filter_chain_from_recommendations(
        self, recommended_filters: List[str]
    ) -> None:
        """Update filter chain based on performance recommendations."""
        filter_mapping = {
            "normalization": "normalization",
            "high_pass_filter": "spectral_enhancement",
            "light_noise_reduction": "noise_reduction",
            "noise_reduction": "noise_reduction",
            "aggressive_noise_reduction": "noise_reduction",
            "spectral_enhancement": "spectral_enhancement",
            "transient_suppression": "spectral_enhancement",
        }

        # Build new filter chain
        new_filter_chain = []
        for filter_name in recommended_filters:
            mapped_name = filter_mapping.get(filter_name, filter_name)
            if mapped_name not in new_filter_chain:
                new_filter_chain.append(mapped_name)

        # Update filter chain
        self.filter_chain = new_filter_chain

        # Initialize stats for new filters
        for filter_name in self.filter_chain:
            if filter_name not in self._filter_stats:
                self._filter_stats[filter_name] = {
                    "calls": 0,
                    "total_time_ms": 0.0,
                    "failures": 0,
                    "last_success": True,
                }

    def _calculate_quality_score(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        filters_applied: bool,
    ) -> float:
        """
        Calculate a quality score for the processed audio.

        Args:
            original_audio: Original audio data
            processed_audio: Processed audio data
            filters_applied: Whether any filters were applied

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not filters_applied:
            return 0.8  # Baseline score for unprocessed audio

        try:
            # Simple quality metrics
            # 1. Check for clipping
            clipping_penalty = 0.0
            if np.max(np.abs(processed_audio)) > 0.95:
                clipping_penalty = 0.2

            # 2. Check for excessive noise reduction (signal loss)
            original_rms = np.sqrt(np.mean(original_audio**2))
            processed_rms = np.sqrt(np.mean(processed_audio**2))

            signal_loss_penalty = 0.0
            if original_rms > 0:
                signal_ratio = processed_rms / original_rms
                if signal_ratio < 0.3:  # Too much signal loss
                    signal_loss_penalty = 0.3
                elif signal_ratio < 0.5:
                    signal_loss_penalty = 0.1

            # 3. Base quality score
            base_score = 0.9

            # Calculate final score
            quality_score = base_score - clipping_penalty - signal_loss_penalty

            return max(0.0, min(1.0, quality_score))

        except Exception:
            # Return neutral score on calculation error
            return 0.7

    def bypass_filter(self, filter_name: str) -> None:
        """
        Bypass a specific filter in the pipeline.

        Args:
            filter_name: Name of the filter to bypass
        """
        self._bypassed_filters.add(filter_name)

    def enable_filter(self, filter_name: str) -> None:
        """
        Re-enable a previously bypassed filter.

        Args:
            filter_name: Name of the filter to re-enable
        """
        self._bypassed_filters.discard(filter_name)
        # Reset failure count when re-enabling
        self._filter_failures[filter_name] = 0

    def set_noise_profile(self, noise_sample: bytes) -> None:
        """
        Set noise profile for noise reduction.

        Args:
            noise_sample: Audio sample containing noise to profile
        """
        try:
            noise_data = np.frombuffer(noise_sample, dtype=np.float32)
            self._noise_reduction.update_noise_profile(noise_data)
        except Exception:
            # Ignore noise profile update failures
            pass

    def reset_adaptive_filters(self) -> None:
        """Reset all adaptive filter states."""
        try:
            self._noise_reduction.reset()
        except AttributeError:
            pass

        try:
            self._audio_normalizer.reset()
        except AttributeError:
            pass

        try:
            self._adaptive_processor.reset()
        except AttributeError:
            pass

    def get_filter_stats(self) -> FilterStats:
        """
        Get statistics about the filtering performance.

        Returns:
            FilterStats containing performance metrics
        """
        # Get performance summary from monitor
        perf_summary = self.performance_monitor.get_performance_summary()

        # Get applied filters (non-bypassed)
        applied_filters = [
            f for f in self.filter_chain if f not in self._bypassed_filters
        ]

        # Calculate noise reduction estimate
        noise_reduction_db = 0.0
        try:
            if self._noise_reduction:
                noise_reduction_db = self._noise_reduction.get_noise_reduction_db()
        except Exception:
            pass

        # Calculate signal enhancement estimate
        signal_enhancement_db = 2.0 if "spectral_enhancement" in applied_filters else 0.0

        return FilterStats(
            noise_reduction_db=noise_reduction_db,
            signal_enhancement_db=signal_enhancement_db,
            processing_latency_ms=perf_summary.get("avg_latency_ms", 0.0),
            filters_applied=applied_filters,
            audio_quality_score=perf_summary.get("avg_quality", 0.8),
        )

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary containing performance metrics and optimization status
        """
        return self.performance_monitor.get_performance_summary()

    def is_performance_degraded(self) -> bool:
        """
        Check if performance is currently degraded.

        Returns:
            True if performance optimization is active
        """
        return self.performance_monitor.is_performance_critical()

    def get_current_complexity_level(self) -> str:
        """
        Get current filter complexity level name.

        Returns:
            Name of current complexity level
        """
        return self.performance_monitor.get_current_complexity_level().name

    def reset(self) -> None:
        """Reset the filter pipeline state."""
        self._bypassed_filters.clear()
        self._filter_failures.clear()
        self._filter_stats.clear()

        # Reset performance monitor
        self.performance_monitor.reset_statistics()

        # Reset individual components
        self.reset_adaptive_filters()

        # Reinitialize components and filter chain
        self._initialize_filter_components()
        self._initialize_filter_chain()
