"""Performance monitoring and optimization for audio filtering pipeline."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil

from ..config import (
    FILTER_CPU_USAGE_THRESHOLD,
    FILTER_LATENCY_CRITICAL_THRESHOLD_MS,
    FILTER_LATENCY_WARNING_THRESHOLD_MS,
    FILTER_MEMORY_USAGE_THRESHOLD_MB,
    FILTER_PROCESSING_TIMEOUT_SEC,
    FILTER_QUALITY_HISTORY_SIZE,
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for audio filtering."""

    cpu_usage_percent: float
    memory_usage_mb: float
    processing_latency_ms: float
    throughput_chunks_per_sec: float
    filter_success_rate: float
    quality_score: float
    timestamp: float


@dataclass
class FilterComplexityLevel:
    """Filter complexity configuration for adaptive processing."""

    name: str
    enabled_filters: List[str]
    max_latency_ms: float
    aggressiveness: float
    description: str


class AudioFilterPerformanceMonitor:
    """
    Performance monitor for audio filtering pipeline.

    Monitors CPU usage, memory consumption, processing latency, and quality metrics
    to enable adaptive filter complexity adjustment and real-time optimization.
    """

    def __init__(self, target_latency_ms: float = 50.0) -> None:
        """
        Initialize performance monitor.

        Args:
            target_latency_ms: Target processing latency in milliseconds
        """
        self.target_latency_ms = target_latency_ms

        # Performance tracking
        self.metrics_history: deque[PerformanceMetrics] = deque(
            maxlen=FILTER_QUALITY_HISTORY_SIZE
        )
        self.processing_times: deque[float] = deque(maxlen=100)
        self.chunk_timestamps: deque[float] = deque(maxlen=100)

        # System monitoring
        self.process = psutil.Process()
        self.cpu_monitor_interval = 1.0  # seconds
        self.last_cpu_check = 0.0
        self.current_cpu_usage = 0.0

        # Filter complexity levels (from minimal to full processing)
        self.complexity_levels = [
            FilterComplexityLevel(
                name="minimal",
                enabled_filters=["normalization"],
                max_latency_ms=10.0,
                aggressiveness=0.2,
                description="Minimal processing - normalization only",
            ),
            FilterComplexityLevel(
                name="light",
                enabled_filters=["normalization", "high_pass_filter"],
                max_latency_ms=20.0,
                aggressiveness=0.3,
                description="Light processing - normalization + high-pass",
            ),
            FilterComplexityLevel(
                name="moderate",
                enabled_filters=[
                    "normalization",
                    "high_pass_filter",
                    "light_noise_reduction",
                ],
                max_latency_ms=35.0,
                aggressiveness=0.4,
                description="Moderate processing - basic noise reduction",
            ),
            FilterComplexityLevel(
                name="standard",
                enabled_filters=[
                    "normalization",
                    "noise_reduction",
                    "spectral_enhancement",
                ],
                max_latency_ms=50.0,
                aggressiveness=0.5,
                description="Standard processing - full pipeline",
            ),
            FilterComplexityLevel(
                name="aggressive",
                enabled_filters=[
                    "normalization",
                    "aggressive_noise_reduction",
                    "spectral_enhancement",
                    "transient_suppression",
                ],
                max_latency_ms=80.0,
                aggressiveness=0.7,
                description="Aggressive processing - maximum quality",
            ),
        ]

        # Current state
        self.current_complexity_level = 3  # Start with "standard"
        self.performance_degradation_active = False
        self.last_adaptation_time = 0.0
        self.adaptation_cooldown = 2.0  # seconds

        # Buffer management
        self.buffer_pool: List[bytearray] = []
        self.max_pool_size = 10
        self.buffer_size = 8192  # bytes

        # Statistics
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.filter_failures = 0
        self.successful_processes = 0

    async def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        asyncio.create_task(self._monitor_system_resources())

    async def _monitor_system_resources(self) -> None:
        """Background task to monitor system resources."""
        while True:
            try:
                # Update CPU usage
                current_time = time.time()
                if current_time - self.last_cpu_check >= self.cpu_monitor_interval:
                    self.current_cpu_usage = self.process.cpu_percent()
                    self.last_cpu_check = current_time

                # Check for performance issues
                await self._check_performance_thresholds()

                await asyncio.sleep(0.5)  # Check every 500ms

            except Exception:
                # Continue monitoring even if there are errors
                await asyncio.sleep(1.0)

    def record_processing_start(self) -> float:
        """
        Record the start of audio chunk processing.

        Returns:
            Timestamp for measuring processing duration
        """
        return time.perf_counter()

    def record_processing_end(
        self,
        start_time: float,
        success: bool = True,
        quality_score: float = 1.0,
        filters_applied: Optional[List[str]] = None,
    ) -> None:
        """
        Record the end of audio chunk processing.

        Args:
            start_time: Start timestamp from record_processing_start()
            success: Whether processing was successful
            quality_score: Quality score of the processing (0.0 to 1.0)
            filters_applied: List of filters that were applied
        """
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        # Update statistics
        self.total_chunks_processed += 1
        self.total_processing_time_ms += processing_time_ms

        if success:
            self.successful_processes += 1
        else:
            self.filter_failures += 1

        # Record processing time
        self.processing_times.append(processing_time_ms)
        self.chunk_timestamps.append(end_time)

        # Calculate throughput
        throughput = self._calculate_throughput()

        # Get current system metrics
        memory_usage_mb = self._get_memory_usage_mb()

        # Create performance metrics
        metrics = PerformanceMetrics(
            cpu_usage_percent=self.current_cpu_usage,
            memory_usage_mb=memory_usage_mb,
            processing_latency_ms=processing_time_ms,
            throughput_chunks_per_sec=throughput,
            filter_success_rate=self._calculate_success_rate(),
            quality_score=quality_score,
            timestamp=end_time,
        )

        # Store metrics
        self.metrics_history.append(metrics)

        # Trigger adaptation if needed
        asyncio.create_task(self._adapt_complexity_if_needed())

    def _calculate_throughput(self) -> float:
        """Calculate current throughput in chunks per second."""
        if len(self.chunk_timestamps) < 2:
            return 0.0

        # Calculate throughput over last 10 chunks or 5 seconds, whichever is shorter
        current_time = time.time()
        recent_timestamps = [
            ts for ts in self.chunk_timestamps if current_time - ts <= 5.0
        ]

        if len(recent_timestamps) < 2:
            return 0.0

        time_span = recent_timestamps[-1] - recent_timestamps[0]
        if time_span > 0:
            return (len(recent_timestamps) - 1) / time_span
        return 0.0

    def _calculate_success_rate(self) -> float:
        """Calculate filter success rate."""
        total_attempts = self.successful_processes + self.filter_failures
        if total_attempts == 0:
            return 1.0
        return self.successful_processes / total_attempts

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0

    async def _check_performance_thresholds(self) -> None:
        """Check if performance thresholds are exceeded."""
        if not self.metrics_history:
            return

        latest_metrics = self.metrics_history[-1]

        # Check CPU usage
        if latest_metrics.cpu_usage_percent > FILTER_CPU_USAGE_THRESHOLD:
            await self._handle_high_cpu_usage()

        # Check memory usage
        if latest_metrics.memory_usage_mb > FILTER_MEMORY_USAGE_THRESHOLD_MB:
            await self._handle_high_memory_usage()

        # Check latency
        if latest_metrics.processing_latency_ms > FILTER_LATENCY_CRITICAL_THRESHOLD_MS:
            await self._handle_high_latency()

    async def _handle_high_cpu_usage(self) -> None:
        """Handle high CPU usage by reducing complexity."""
        if not self.performance_degradation_active:
            self.performance_degradation_active = True
            await self._reduce_complexity()

    async def _handle_high_memory_usage(self) -> None:
        """Handle high memory usage."""
        # Clear buffer pool to free memory
        self.buffer_pool.clear()

        # Reduce complexity if memory usage is critical
        if not self.performance_degradation_active:
            self.performance_degradation_active = True
            await self._reduce_complexity()

    async def _handle_high_latency(self) -> None:
        """Handle high processing latency."""
        if not self.performance_degradation_active:
            self.performance_degradation_active = True
            await self._reduce_complexity()

    async def _adapt_complexity_if_needed(self) -> None:
        """Adapt filter complexity based on performance metrics."""
        current_time = time.time()

        # Respect cooldown period
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return

        if len(self.metrics_history) < 5:
            return  # Need enough data for adaptation

        # Calculate recent performance averages
        recent_metrics = list(self.metrics_history)[-5:]
        avg_latency = sum(m.processing_latency_ms for m in recent_metrics) / len(
            recent_metrics
        )
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_quality = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)

        # Determine if we should increase or decrease complexity
        should_reduce = (
            avg_latency > FILTER_LATENCY_WARNING_THRESHOLD_MS
            or avg_cpu > FILTER_CPU_USAGE_THRESHOLD * 0.8
        )

        should_increase = (
            avg_latency < self.target_latency_ms * 0.7
            and avg_cpu < FILTER_CPU_USAGE_THRESHOLD * 0.5
            and avg_quality > 0.8
            and not self.performance_degradation_active
        )

        if should_reduce:
            await self._reduce_complexity()
        elif should_increase:
            await self._increase_complexity()

        self.last_adaptation_time = current_time

    async def _reduce_complexity(self) -> None:
        """Reduce filter complexity to improve performance."""
        if self.current_complexity_level > 0:
            self.current_complexity_level -= 1
            self.performance_degradation_active = True

    async def _increase_complexity(self) -> None:
        """Increase filter complexity for better quality."""
        if self.current_complexity_level < len(self.complexity_levels) - 1:
            self.current_complexity_level += 1
            # Reset degradation flag if we're increasing complexity
            if self.current_complexity_level >= 2:  # "moderate" or higher
                self.performance_degradation_active = False

    def get_current_complexity_level(self) -> FilterComplexityLevel:
        """
        Get the current filter complexity level.

        Returns:
            Current FilterComplexityLevel configuration
        """
        return self.complexity_levels[self.current_complexity_level]

    def get_recommended_filters(self) -> List[str]:
        """
        Get list of recommended filters based on current performance.

        Returns:
            List of filter names that should be enabled
        """
        return self.get_current_complexity_level().enabled_filters

    def get_recommended_aggressiveness(self) -> float:
        """
        Get recommended aggressiveness level.

        Returns:
            Aggressiveness value (0.0 to 1.0)
        """
        return self.get_current_complexity_level().aggressiveness

    def get_recommended_max_latency(self) -> float:
        """
        Get recommended maximum latency.

        Returns:
            Maximum latency in milliseconds
        """
        return self.get_current_complexity_level().max_latency_ms

    def get_buffer_from_pool(self, size: int) -> bytearray:
        """
        Get a buffer from the pool for memory-efficient processing.

        Args:
            size: Required buffer size in bytes

        Returns:
            Reusable buffer
        """
        # Try to reuse existing buffer
        for i, buffer in enumerate(self.buffer_pool):
            if len(buffer) >= size:
                return self.buffer_pool.pop(i)

        # Create new buffer if none available
        return bytearray(size)

    def return_buffer_to_pool(self, buffer: bytearray) -> None:
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: Buffer to return to pool
        """
        if len(self.buffer_pool) < self.max_pool_size:
            # Clear buffer and add to pool
            buffer[:] = b"\x00" * len(buffer)
            self.buffer_pool.append(buffer)

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get performance summary statistics.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.metrics_history:
            return {}

        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements

        return {
            "avg_latency_ms": sum(m.processing_latency_ms for m in recent_metrics)
            / len(recent_metrics),
            "avg_cpu_percent": sum(m.cpu_usage_percent for m in recent_metrics)
            / len(recent_metrics),
            "avg_memory_mb": sum(m.memory_usage_mb for m in recent_metrics)
            / len(recent_metrics),
            "avg_throughput": sum(m.throughput_chunks_per_sec for m in recent_metrics)
            / len(recent_metrics),
            "success_rate": self._calculate_success_rate(),
            "avg_quality": sum(m.quality_score for m in recent_metrics)
            / len(recent_metrics),
            "total_chunks": self.total_chunks_processed,
            "current_complexity": self.current_complexity_level,
            "performance_degradation": self.performance_degradation_active,
        }

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.metrics_history.clear()
        self.processing_times.clear()
        self.chunk_timestamps.clear()
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.filter_failures = 0
        self.successful_processes = 0
        self.performance_degradation_active = False
        self.current_complexity_level = 3  # Reset to "standard"

    def is_performance_critical(self) -> bool:
        """
        Check if performance is in critical state.

        Returns:
            True if performance is critically degraded
        """
        if not self.metrics_history:
            return False

        latest = self.metrics_history[-1]
        return (
            latest.processing_latency_ms > FILTER_LATENCY_CRITICAL_THRESHOLD_MS
            or latest.cpu_usage_percent > FILTER_CPU_USAGE_THRESHOLD * 1.2
            or latest.memory_usage_mb > FILTER_MEMORY_USAGE_THRESHOLD_MB * 1.5
        )

    def should_bypass_filtering(self) -> bool:
        """
        Determine if filtering should be bypassed due to performance issues.

        Returns:
            True if filtering should be bypassed
        """
        return (
            self.is_performance_critical()
            and self.current_complexity_level == 0
            and self.performance_degradation_active
        )
