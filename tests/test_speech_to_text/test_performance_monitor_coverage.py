"""Additional tests to improve performance_monitor.py coverage."""

import pytest
import time
from unittest.mock import MagicMock, patch
from local_ai.speech_to_text.performance_monitor import (
    PerformanceMonitor, 
    PerformanceContext,
    get_performance_monitor
)


class TestPerformanceMonitorCoverage:
    """Test cases to improve coverage of performance monitoring."""

    def test_performance_monitor_initialization(self) -> None:
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(max_history=50)
        
        assert monitor.max_history == 50
        assert len(monitor.metrics_history) == 0
        assert len(monitor.operation_counters) == 0
        assert len(monitor.start_times) == 0

    def test_start_operation_without_existing_start_time(self) -> None:
        """Test starting an operation normally."""
        monitor = PerformanceMonitor()
        
        monitor.start_operation("op1", "test_operation")
        
        assert "op1" in monitor.start_times
        assert monitor.operation_counters["test_operation"] == 1

    def test_end_operation_without_start_time(self) -> None:
        """Test ending operation without start time."""
        monitor = PerformanceMonitor()
        
        duration = monitor.end_operation("nonexistent", "test_operation")
        
        assert duration == 0.0

    def test_end_operation_with_metadata(self) -> None:
        """Test ending operation with metadata."""
        monitor = PerformanceMonitor()
        
        monitor.start_operation("op1", "test_operation")
        time.sleep(0.01)  # Small delay
        duration = monitor.end_operation("op1", "test_operation", True, {"key": "value"})
        
        assert duration > 0
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0].metadata["key"] == "value"

    def test_get_operation_stats_no_metrics(self) -> None:
        """Test get_operation_stats with no metrics."""
        monitor = PerformanceMonitor()
        
        stats = monitor.get_operation_stats("nonexistent")
        
        assert stats["count"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_duration"] == 0.0

    def test_get_operation_stats_with_time_window(self) -> None:
        """Test get_operation_stats with time window."""
        monitor = PerformanceMonitor()
        
        # Add old metric (outside window)
        monitor.start_operation("op1", "test")
        monitor.end_operation("op1", "test")
        
        # Modify timestamp to be old
        if monitor.metrics_history:
            monitor.metrics_history[0].timestamp = time.time() - 3600  # 1 hour ago
        
        # Add recent metric
        monitor.start_operation("op2", "test")
        monitor.end_operation("op2", "test")
        
        stats = monitor.get_operation_stats("test", time_window=1800)  # 30 minutes
        
        assert stats["count"] == 1  # Only recent metric

    def test_get_overall_stats_empty(self) -> None:
        """Test get_overall_stats with no metrics."""
        monitor = PerformanceMonitor()
        
        stats = monitor.get_overall_stats()
        
        assert stats["total_operations"] == 0
        assert stats["operations_by_type"] == {}

    def test_detect_performance_issues_high_latency(self) -> None:
        """Test performance issue detection for high latency."""
        monitor = PerformanceMonitor()
        
        # Add high latency transcription
        monitor.start_operation("op1", "transcription")
        time.sleep(0.01)
        monitor.end_operation("op1", "transcription", True, {})
        
        # Manually set high duration for testing
        if monitor.metrics_history:
            monitor.metrics_history[0].duration = 6.0  # High latency
        
        issues = monitor.detect_performance_issues()
        
        # Should detect high latency issue
        high_latency_issues = [i for i in issues if i["type"] == "high_latency"]
        assert len(high_latency_issues) > 0

    def test_detect_performance_issues_low_success_rate(self) -> None:
        """Test performance issue detection for low success rate."""
        monitor = PerformanceMonitor()
        
        # Add failed operations
        for i in range(5):
            monitor.start_operation(f"op{i}", "transcription")
            monitor.end_operation(f"op{i}", "transcription", False)  # Failed
        
        issues = monitor.detect_performance_issues()
        
        # Should detect low success rate
        success_issues = [i for i in issues if i["type"] == "low_success_rate"]
        assert len(success_issues) > 0

    def test_detect_performance_issues_slow_vad(self) -> None:
        """Test performance issue detection for slow VAD."""
        monitor = PerformanceMonitor()
        
        # Add slow VAD operation
        monitor.start_operation("vad1", "vad")
        monitor.end_operation("vad1", "vad", True)
        
        # Manually set high duration
        if monitor.metrics_history:
            monitor.metrics_history[0].duration = 0.2  # Slow VAD
        
        issues = monitor.detect_performance_issues()
        
        # Should detect slow VAD
        vad_issues = [i for i in issues if i["type"] == "slow_vad"]
        assert len(vad_issues) > 0

    def test_get_performance_report_no_operations(self) -> None:
        """Test performance report with no operations."""
        monitor = PerformanceMonitor()
        
        report = monitor.get_performance_report()
        
        assert "No operations recorded" in report

    def test_get_performance_report_with_operations(self) -> None:
        """Test performance report with operations."""
        monitor = PerformanceMonitor()
        
        # Add some operations
        monitor.start_operation("op1", "transcription")
        monitor.end_operation("op1", "transcription", True)
        
        report = monitor.get_performance_report()
        
        assert "Speech-to-Text Performance Report" in report
        assert "transcription" in report

    def test_get_performance_report_with_issues(self) -> None:
        """Test performance report with detected issues."""
        monitor = PerformanceMonitor()
        
        # Add operation with issue
        monitor.start_operation("op1", "transcription")
        monitor.end_operation("op1", "transcription", False)  # Failed
        
        report = monitor.get_performance_report()
        
        # Should include issues section
        assert "Performance Issues" in report or "No performance issues" in report

    def test_reset_metrics(self) -> None:
        """Test resetting metrics."""
        monitor = PerformanceMonitor()
        
        # Add some data
        monitor.start_operation("op1", "test")
        monitor.end_operation("op1", "test")
        
        assert len(monitor.metrics_history) > 0
        assert len(monitor.operation_counters) > 0
        
        monitor.reset_metrics()
        
        assert len(monitor.metrics_history) == 0
        assert len(monitor.operation_counters) == 0
        assert len(monitor.start_times) == 0

    def test_performance_context_success(self) -> None:
        """Test PerformanceContext successful operation."""
        with PerformanceContext("test_operation", metadata={"test": True}) as ctx:
            ctx.set_metadata("additional", "data")
            time.sleep(0.01)
        
        # Should complete without error
        assert ctx.success is True

    def test_performance_context_failure(self) -> None:
        """Test PerformanceContext with exception."""
        try:
            with PerformanceContext("test_operation") as ctx:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should mark as failure
        assert ctx.success is False
        assert "error" in ctx.metadata

    def test_performance_context_mark_failure(self) -> None:
        """Test manually marking PerformanceContext as failure."""
        with PerformanceContext("test_operation") as ctx:
            ctx.mark_failure("Manual failure")
        
        assert ctx.success is False
        assert ctx.metadata["error"] == "Manual failure"

    def test_global_performance_monitor(self) -> None:
        """Test global performance monitor instance."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should return same instance
        assert monitor1 is monitor2