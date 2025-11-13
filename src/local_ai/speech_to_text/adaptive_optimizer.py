"""Adaptive optimization based on runtime performance."""

from typing import TYPE_CHECKING, Any

from .config import (
    HIGH_CPU_THRESHOLD,
    HIGH_LATENCY_THRESHOLD,
    MINIMUM_CHUNK_SIZE,
    MINIMUM_PROCESSING_INTERVAL,
    OPT_ADAPTIVE_INTERVAL_MAX,
    OPT_ADAPTIVE_LATENCY_THRESHOLD,
    OPT_ADAPTIVE_MIN_HISTORY,
    OPT_ADAPTIVE_RECENT_WINDOW,
    OPT_ADAPTIVE_SILENCE_REDUCTION,
    OPT_MODEL_SIZE_LARGE,
    OPT_MODEL_SIZE_MEDIUM,
    OPT_MODEL_SIZE_SMALL,
    OPT_USE_DISTILLED_MODELS,
    OPT_USE_ENGLISH_ONLY_MODEL,
    PERFORMANCE_HISTORY_SIZE,
    RESOURCE_CHUNK_SIZE,
)
from .logging_utils import get_logger
from .model_selector import apply_model_optimizations

if TYPE_CHECKING:
    from .performance_optimizer import PerformanceOptimizer

logger = get_logger(__name__)


class AdaptiveOptimizer:
    """Dynamically adapts configuration based on runtime performance."""

    def __init__(self, base_optimizer: "PerformanceOptimizer"):
        """Initialize adaptive optimizer."""
        self.base_optimizer = base_optimizer
        self.performance_history: list[dict[str, Any]] = []
        self.current_config = base_optimizer.optimized_config.copy()
        self.adaptation_count = 0

    def record_performance(
        self, latency: float, cpu_usage: float, memory_usage: float
    ) -> None:
        """Record performance metrics for adaptation."""
        self.performance_history.append(
            {
                "latency": latency,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "timestamp": __import__("time").time(),
            }
        )

        # Keep only recent history (last N measurements)
        if len(self.performance_history) > PERFORMANCE_HISTORY_SIZE:
            self.performance_history.pop(0)

    def should_adapt(self) -> bool:
        """Determine if configuration should be adapted."""
        if len(self.performance_history) < OPT_ADAPTIVE_MIN_HISTORY:
            return False

        # Check if performance is consistently poor
        recent_latencies = [
            p["latency"] for p in self.performance_history[-OPT_ADAPTIVE_RECENT_WINDOW:]
        ]
        recent_cpu = [
            p["cpu_usage"] for p in self.performance_history[-OPT_ADAPTIVE_RECENT_WINDOW:]
        ]

        avg_latency = sum(recent_latencies) / len(recent_latencies)
        avg_cpu = sum(recent_cpu) / len(recent_cpu)

        # Adapt if latency is too high or CPU usage is too high
        return bool(avg_latency > HIGH_LATENCY_THRESHOLD or avg_cpu > HIGH_CPU_THRESHOLD)

    def adapt_configuration(self) -> dict[str, Any]:
        """Adapt configuration based on performance history."""
        if not self.should_adapt():
            return self.current_config

        self.adaptation_count += 1
        logger.debug(f"Adapting configuration (adaptation #{self.adaptation_count})")

        # Get recent performance metrics
        recent_metrics = self.performance_history[-OPT_ADAPTIVE_RECENT_WINDOW:]
        avg_latency = sum(p["latency"] for p in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(p["cpu_usage"] for p in recent_metrics) / len(recent_metrics)

        # Adapt based on performance issues
        if avg_latency > OPT_ADAPTIVE_LATENCY_THRESHOLD:
            # High latency - optimize for speed
            logger.debug("High latency detected, optimizing for speed")
            self.current_config["chunk_size"] = max(
                MINIMUM_CHUNK_SIZE, self.current_config["chunk_size"] // 2
            )
            self.current_config["processing_interval"] = max(
                MINIMUM_PROCESSING_INTERVAL,
                self.current_config["processing_interval"] / 2,
            )
            self.current_config["max_silence_duration"] = max(
                OPT_ADAPTIVE_SILENCE_REDUCTION,
                self.current_config["max_silence_duration"]
                - OPT_ADAPTIVE_SILENCE_REDUCTION,
            )

        if avg_cpu > HIGH_CPU_THRESHOLD:
            # High CPU usage - reduce processing load
            logger.debug("High CPU usage detected, reducing processing load")
            self.current_config["chunk_size"] = min(
                RESOURCE_CHUNK_SIZE, self.current_config["chunk_size"] * 2
            )
            self.current_config["processing_interval"] = min(
                OPT_ADAPTIVE_INTERVAL_MAX,
                self.current_config["processing_interval"] * 2,
            )

            # Switch to smaller model if using larger one
            current_model = self.current_config["whisper_model_size"]
            # Handle distilled, .en, and base model names
            base_model = (
                current_model.replace("distil-", "").replace(".en", "").replace("-v3", "")
            )

            if base_model == OPT_MODEL_SIZE_LARGE:
                self.current_config["whisper_model_size"] = apply_model_optimizations(
                    OPT_MODEL_SIZE_MEDIUM,
                    OPT_USE_ENGLISH_ONLY_MODEL,
                    OPT_USE_DISTILLED_MODELS,
                )
            elif base_model == OPT_MODEL_SIZE_MEDIUM:
                self.current_config["whisper_model_size"] = apply_model_optimizations(
                    OPT_MODEL_SIZE_SMALL,
                    OPT_USE_ENGLISH_ONLY_MODEL,
                    OPT_USE_DISTILLED_MODELS,
                )

        return self.current_config

    def get_current_config(self) -> dict[str, Any]:
        """Get current adapted configuration."""
        return self.current_config.copy()
