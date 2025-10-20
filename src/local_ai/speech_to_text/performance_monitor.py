"""Performance monitoring for speech-to-text pipeline."""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .config import PERFORMANCE_HISTORY_SIZE, VAD_SLOW_THRESHOLD
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""

    timestamp: float
    operation: str
    duration: float
    success: bool
    metadata: dict[str, Any]


class PerformanceMonitor:
    """Monitors and tracks performance metrics for the speech-to-text pipeline."""

    def __init__(self, max_history: int = None):
        """
        Initialize performance monitor.

        Args:
            max_history: Maximum number of metrics to keep in history
        """
        if max_history is None:
            max_history = PERFORMANCE_HISTORY_SIZE
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_counters: dict[str, int] = {}
        self.start_times: dict[str, float] = {}

    def start_operation(self, operation_id: str, operation_type: str) -> None:
        """
        Start timing an operation.

        Args:
            operation_id: Unique identifier for this operation instance
            operation_type: Type of operation (e.g., "transcription", "vad")
        """
        self.start_times[operation_id] = time.time()
        self.operation_counters[operation_type] = (
            self.operation_counters.get(operation_type, 0) + 1
        )

    def end_operation(
        self,
        operation_id: str,
        operation_type: str,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        End timing an operation and record metrics.

        Args:
            operation_id: Unique identifier for this operation instance
            operation_type: Type of operation
            success: Whether the operation succeeded
            metadata: Additional metadata about the operation

        Returns:
            Duration of the operation in seconds
        """
        if operation_id not in self.start_times:
            logger.warning(f"No start time found for operation {operation_id}")
            return 0.0

        duration = time.time() - self.start_times[operation_id]
        del self.start_times[operation_id]

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            operation=operation_type,
            duration=duration,
            success=success,
            metadata=metadata or {},
        )

        self.metrics_history.append(metrics)

        logger.debug(
            f"{operation_type} completed in {duration:.3f}s (success: {success})"
        )
        return duration

    def get_operation_stats(
        self, operation_type: str, time_window: float | None = None
    ) -> dict[str, Any]:
        """
        Get statistics for a specific operation type.

        Args:
            operation_type: Type of operation to analyze
            time_window: Only consider metrics within this many seconds (None for all)

        Returns:
            Dictionary with statistics
        """
        current_time = time.time()
        relevant_metrics = []

        for metric in self.metrics_history:
            if metric.operation == operation_type:
                if (
                    time_window is None
                    or (current_time - metric.timestamp) <= time_window
                ):
                    relevant_metrics.append(metric)

        if not relevant_metrics:
            return {
                "count": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
            }

        durations = [m.duration for m in relevant_metrics]
        successes = [m.success for m in relevant_metrics]

        return {
            "count": len(relevant_metrics),
            "success_rate": sum(successes) / len(successes),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
        }

    def get_overall_stats(self, time_window: float | None = None) -> dict[str, Any]:
        """
        Get overall performance statistics.

        Args:
            time_window: Only consider metrics within this many seconds (None for all)

        Returns:
            Dictionary with overall statistics
        """
        current_time = time.time()
        relevant_metrics = []

        for metric in self.metrics_history:
            if time_window is None or (current_time - metric.timestamp) <= time_window:
                relevant_metrics.append(metric)

        if not relevant_metrics:
            return {"total_operations": 0, "operations_by_type": {}}

        # Group by operation type
        operations_by_type = {}
        for metric in relevant_metrics:
            if metric.operation not in operations_by_type:
                operations_by_type[metric.operation] = []
            operations_by_type[metric.operation].append(metric)

        # Calculate stats for each operation type
        type_stats = {}
        for op_type, metrics in operations_by_type.items():
            durations = [m.duration for m in metrics]
            successes = [m.success for m in metrics]

            type_stats[op_type] = {
                "count": len(metrics),
                "success_rate": sum(successes) / len(successes),
                "avg_duration": sum(durations) / len(durations),
                "total_duration": sum(durations),
            }

        return {
            "total_operations": len(relevant_metrics),
            "operations_by_type": type_stats,
            "time_window": time_window,
        }

    def detect_performance_issues(self) -> list[dict[str, Any]]:
        """
        Detect potential performance issues based on metrics.

        Returns:
            List of detected issues with descriptions and recommendations
        """
        issues = []

        # Check recent transcription performance (last 60 seconds)
        transcription_stats = self.get_operation_stats("transcription", time_window=60.0)

        if transcription_stats["count"] > 0:
            # Check for high latency
            if transcription_stats["avg_duration"] > 5.0:
                issues.append(
                    {
                        "type": "high_latency",
                        "severity": "warning",
                        "description": (
                            f"Latency: {transcription_stats['avg_duration']:.2f}s"
                        ),
                        "recommendation": "Consider using a smaller model",
                    }
                )

            # Check for low success rate
            if transcription_stats["success_rate"] < 0.9:
                issues.append(
                    {
                        "type": "low_success_rate",
                        "severity": "error",
                        "description": (
                            f"Success rate: {transcription_stats['success_rate']:.1%}"
                        ),
                        "recommendation": "Check audio input or model availability",
                    }
                )

        # Check VAD performance
        vad_stats = self.get_operation_stats("vad", time_window=60.0)
        if vad_stats["count"] > 0 and vad_stats["avg_duration"] > VAD_SLOW_THRESHOLD:
            issues.append(
                {
                    "type": "slow_vad",
                    "severity": "info",
                    "description": f"Slow VAD: {vad_stats['avg_duration']:.3f}s",
                    "recommendation": "Consider adjusting VAD frame duration",
                }
            )

        return issues

    def get_performance_report(self) -> str:
        """
        Generate a human-readable performance report.

        Returns:
            Formatted performance report string
        """
        overall_stats = self.get_overall_stats(time_window=300.0)  # Last 5 minutes
        issues = self.detect_performance_issues()

        report = ["Speech-to-Text Performance Report", "=" * 40]

        if overall_stats["total_operations"] == 0:
            report.append("No operations recorded in the last 5 minutes.")
            return "\n".join(report)

        report.append(
            f"Total operations (last 5 min): {overall_stats['total_operations']}"
        )
        report.append("")

        # Operation type breakdown
        report.append("Operations by type:")
        for op_type, stats in overall_stats["operations_by_type"].items():
            report.append(f"  {op_type}:")
            report.append(f"    Count: {stats['count']}")
            report.append(f"    Success rate: {stats['success_rate']:.1%}")
            report.append(f"    Avg duration: {stats['avg_duration']:.3f}s")
            report.append(f"    Total time: {stats['total_duration']:.2f}s")

        # Performance issues
        if issues:
            report.append("")
            report.append("Performance Issues:")
            for issue in issues:
                report.append(f"  [{issue['severity'].upper()}] {issue['description']}")
                report.append(f"    Recommendation: {issue['recommendation']}")
        else:
            report.append("")
            report.append("No performance issues detected.")

        return "\n".join(report)

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.metrics_history.clear()
        self.operation_counters.clear()
        self.start_times.clear()
        logger.debug("Performance metrics reset")


# Global performance monitor instance
_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


class PerformanceContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        operation_type: str,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize performance context.

        Args:
            operation_type: Type of operation being timed
            operation_id: Unique ID for this operation (auto-generated if None)
            metadata: Additional metadata to record
        """
        self.operation_type = operation_type
        self.operation_id = (
            operation_id or f"{operation_type}_{int(time.time() * 1000000)}"
        )
        self.metadata = metadata or {}
        self.monitor = get_performance_monitor()
        self.success = True

    def __enter__(self):
        """Start timing the operation."""
        self.monitor.start_operation(self.operation_id, self.operation_type)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing the operation."""
        if exc_type is not None:
            self.success = False
            self.metadata["error"] = str(exc_val)

        self.monitor.end_operation(
            self.operation_id, self.operation_type, self.success, self.metadata
        )

    def set_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation."""
        self.metadata[key] = value

    def mark_failure(self, error_message: str = "") -> None:
        """Mark the operation as failed."""
        self.success = False
        if error_message:
            self.metadata["error"] = error_message
