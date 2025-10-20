"""Performance optimizations for speech-to-text pipeline."""

from typing import Any

from .config import (
    ACCURACY_BUFFER_SIZE,
    ACCURACY_MAX_SILENCE_DURATION,
    ACCURACY_MIN_SPEECH_DURATION,
    BYTES_PER_KB,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_AUDIO_BUFFER_SIZE,
    DEFAULT_MAX_SILENCE_DURATION,
    DEFAULT_MIN_SPEECH_DURATION,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TRANSCRIPTION_TIMEOUT,
    HIGH_CPU_THRESHOLD,
    HIGH_GPU_MEMORY_GB,
    HIGH_LATENCY_THRESHOLD,
    HIGH_MEMORY_BUFFER_SIZE,
    HIGH_MEMORY_GB,
    LATENCY_BUFFER_SIZE,
    LATENCY_CHUNK_SIZE,
    LATENCY_MAX_SILENCE_DURATION,
    LATENCY_MIN_SPEECH_DURATION,
    LATENCY_VAD_FRAME_DURATION,
    MINIMUM_CHUNK_SIZE,
    MINIMUM_PROCESSING_INTERVAL,
    PERFORMANCE_HISTORY_SIZE,
    RESOURCE_CHUNK_SIZE,
    RESOURCE_PROCESSING_INTERVAL,
    ULTRA_GPU_MEMORY_GB,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION,
)
from .logging_utils import get_logger
from .optimization_cache import get_optimization_cache

logger = get_logger(__name__)


class PerformanceOptimizer:
    """Optimizes speech-to-text pipeline performance based on system capabilities."""

    def __init__(self, use_cache: bool = True, force_cpu: bool = False):
        """
        Initialize the performance optimizer.

        Args:
            use_cache: Whether to use cached optimization data
            force_cpu: Whether to force CPU-only mode (disable GPU/CUDA)
        """
        self.use_cache = use_cache
        self.force_cpu = force_cpu
        self.cache = get_optimization_cache() if use_cache else None
        self.system_info = self._detect_system_capabilities()
        self.optimized_config = self._generate_optimized_config()

    def _detect_system_capabilities(self) -> dict[str, Any]:
        """Detect system capabilities for optimization with caching."""
        # Try to get from cache first
        if self.use_cache and self.cache:
            cached_capabilities = self.cache.get_cached_system_capabilities()
            if cached_capabilities:
                return cached_capabilities

        # Detect capabilities
        import os
        import platform

        capabilities = {
            "cpu_count": os.cpu_count() or 1,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "has_gpu": False,
            "memory_gb": 4,  # Default conservative estimate
        }

        # Check if GPU should be forced off
        if self.force_cpu:
            logger.debug("GPU detection disabled - force CPU mode requested")
            capabilities["has_gpu"] = False
        else:
            # Try to detect GPU capabilities
            capabilities["has_gpu"] = self._validate_cuda_availability()
            if capabilities["has_gpu"]:
                # Try to get GPU memory info
                try:
                    import torch

                    if torch.cuda.is_available():
                        gpu_memory_bytes = torch.cuda.get_device_properties(
                            0
                        ).total_memory
                        capabilities["gpu_memory_gb"] = gpu_memory_bytes / (
                            BYTES_PER_KB**3
                        )
                        logger.debug(
                            f"GPU detected: {capabilities['gpu_memory_gb']:.1f}GB memory"
                        )
                except Exception as e:
                    logger.debug(f"Could not get GPU memory info: {e}")
                    capabilities["gpu_memory_gb"] = 0

        # Try to detect memory
        try:
            import psutil

            capabilities["memory_gb"] = psutil.virtual_memory().total / (BYTES_PER_KB**3)
        except ImportError:
            pass

        logger.debug(f"Detected system capabilities: {capabilities}")

        # Cache the results
        if self.use_cache and self.cache:
            self.cache.cache_system_capabilities(capabilities)

        return capabilities

    def _validate_cuda_availability(self) -> bool:
        """
        Validate that CUDA is actually working, not just available.

        Returns:
            True if CUDA is available and working, False otherwise
        """
        try:
            import torch

            # First check if torch thinks CUDA is available
            if not torch.cuda.is_available():
                logger.debug("CUDA not available according to PyTorch")
                return False

            # Try to actually use CUDA to validate it works
            logger.debug("Validating CUDA functionality...")

            # Test basic CUDA operations
            device = torch.device("cuda:0")
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            result = test_tensor * 2
            result_cpu = result.cpu()

            # Verify the operation worked
            expected = torch.tensor([2.0, 4.0, 6.0])
            if not torch.allclose(result_cpu, expected):
                logger.error("CUDA validation failed - tensor operations incorrect")
                return False

            logger.debug("CUDA validation successful")
            return True

        except ImportError:
            logger.debug("PyTorch not available - CUDA disabled")
            return False
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "cudnn" in error_msg or "gpu" in error_msg:
                logger.warning(f"CUDA validation failed with runtime error: {e}")
                logger.debug("Falling back to CPU-only mode due to CUDA issues")
                return False
            # Re-raise if it's not a CUDA-related error
            raise
        except Exception as e:
            logger.warning(f"CUDA validation failed with unexpected error: {e}")
            logger.debug("Falling back to CPU-only mode for safety")
            return False

    def _generate_optimized_config(self) -> dict[str, Any]:
        """Generate optimized configuration based on system capabilities."""
        config = {
            # Audio processing optimizations
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "buffer_size": DEFAULT_BUFFER_SIZE,
            # VAD optimizations
            "vad_aggressiveness": VAD_AGGRESSIVENESS,
            "vad_frame_duration": VAD_FRAME_DURATION,
            "min_speech_duration": DEFAULT_MIN_SPEECH_DURATION,
            "max_silence_duration": DEFAULT_MAX_SILENCE_DURATION,
            # Transcription optimizations
            "whisper_model_size": "small",  # Default
            "compute_type": "int8",  # CPU optimization
            "device": "cpu",  # Default to CPU for stability
            "max_audio_buffer_size": DEFAULT_MAX_AUDIO_BUFFER_SIZE,
            "transcription_timeout": DEFAULT_TRANSCRIPTION_TIMEOUT,
            # Pipeline optimizations
            "processing_interval": 0.01,  # seconds between processing cycles
            "max_concurrent_transcriptions": 1,  # Prevent overload
        }

        # Optimize based on system capabilities
        if self.system_info["cpu_count"] >= 4:
            config["chunk_size"] = (
                LATENCY_CHUNK_SIZE  # Smaller chunks for better responsiveness
            )
            config["processing_interval"] = 0.005  # Faster processing

        if self.system_info["memory_gb"] >= HIGH_MEMORY_GB:
            config["buffer_size"] = HIGH_MEMORY_BUFFER_SIZE  # ~10 seconds buffer
            config["max_audio_buffer_size"] = 15

        # Configure device based on GPU availability and force_cpu setting
        if self.force_cpu or not self.system_info["has_gpu"]:
            if self.force_cpu:
                logger.debug(
                    "Using CPU-only mode for Whisper transcription (force CPU requested)"
                )
            else:
                logger.debug(
                    "Using CPU-only mode for Whisper transcription (no GPU available)"
                )
            config["device"] = "cpu"
            config["compute_type"] = "int8"
        else:
            logger.debug("Using GPU acceleration for Whisper transcription")
            config["device"] = "cuda"
            config["compute_type"] = "float16"  # Better for GPU

            # Adjust model size based on GPU memory
            gpu_memory = self.system_info.get("gpu_memory_gb", 0)
            if gpu_memory >= ULTRA_GPU_MEMORY_GB:
                config["whisper_model_size"] = "large"
            elif gpu_memory >= HIGH_GPU_MEMORY_GB:
                config["whisper_model_size"] = "medium"
            else:
                config["whisper_model_size"] = "small"

        # Optimize CPU performance based on system capabilities
        if self.system_info["cpu_count"] >= 8:
            config["whisper_model_size"] = (
                "small"  # Can handle slightly larger model with more CPU cores
            )
        elif self.system_info["cpu_count"] >= 4:
            config["whisper_model_size"] = (
                "tiny"  # Use smallest model for better performance on limited cores
            )

        # Platform-specific optimizations
        if self.system_info["platform"] == "Linux":
            config["vad_aggressiveness"] = min(
                3, VAD_AGGRESSIVENESS + 1
            )  # More aggressive on Linux

        logger.debug(f"Generated optimized config: {config}")
        return config

    def get_optimized_audio_config(self) -> dict[str, Any]:
        """Get optimized audio capture configuration."""
        return {
            "sample_rate": self.optimized_config["sample_rate"],
            "chunk_size": self.optimized_config["chunk_size"],
        }

    def get_optimized_vad_config(self) -> dict[str, Any]:
        """Get optimized VAD configuration."""
        return {
            "sample_rate": self.optimized_config["sample_rate"],
            "frame_duration": self.optimized_config["vad_frame_duration"],
            "aggressiveness": self.optimized_config["vad_aggressiveness"],
        }

    def get_optimized_transcriber_config(self) -> dict[str, Any]:
        """Get optimized transcriber configuration."""
        return {
            "model_size": self.optimized_config["whisper_model_size"],
            "device": self.optimized_config["device"],
            "compute_type": self.optimized_config["compute_type"],
        }

    def get_optimized_pipeline_config(self) -> dict[str, Any]:
        """Get optimized pipeline configuration."""
        return {
            "buffer_size": self.optimized_config["buffer_size"],
            "min_speech_duration": self.optimized_config["min_speech_duration"],
            "max_silence_duration": self.optimized_config["max_silence_duration"],
            "processing_interval": self.optimized_config["processing_interval"],
            "max_audio_buffer_size": self.optimized_config["max_audio_buffer_size"],
            "transcription_timeout": self.optimized_config["transcription_timeout"],
            "max_concurrent_transcriptions": self.optimized_config[
                "max_concurrent_transcriptions"
            ],
        }

    def optimize_for_latency(self) -> dict[str, Any]:
        """Get configuration optimized for low latency."""
        latency_config = self.optimized_config.copy()

        # Reduce buffer sizes for faster response
        latency_config["chunk_size"] = LATENCY_CHUNK_SIZE
        latency_config["buffer_size"] = LATENCY_BUFFER_SIZE
        latency_config["min_speech_duration"] = LATENCY_MIN_SPEECH_DURATION
        latency_config["max_silence_duration"] = LATENCY_MAX_SILENCE_DURATION
        latency_config["processing_interval"] = 0.005  # More frequent processing
        latency_config["vad_frame_duration"] = LATENCY_VAD_FRAME_DURATION

        logger.debug("Applied latency optimizations")
        return latency_config

    def optimize_for_accuracy(self) -> dict[str, Any]:
        """Get configuration optimized for accuracy."""
        accuracy_config = self.optimized_config.copy()

        # Increase buffer sizes for better context
        accuracy_config["buffer_size"] = ACCURACY_BUFFER_SIZE
        accuracy_config["min_speech_duration"] = ACCURACY_MIN_SPEECH_DURATION
        accuracy_config["max_silence_duration"] = ACCURACY_MAX_SILENCE_DURATION
        accuracy_config["vad_aggressiveness"] = max(
            1, VAD_AGGRESSIVENESS - 1
        )  # Less aggressive

        # Use better model if system supports it
        if self.system_info.get("gpu_memory_gb", 0) >= HIGH_GPU_MEMORY_GB:
            accuracy_config["whisper_model_size"] = "medium"
        if self.system_info.get("gpu_memory_gb", 0) >= ULTRA_GPU_MEMORY_GB:
            accuracy_config["whisper_model_size"] = "large"

        logger.debug("Applied accuracy optimizations")
        return accuracy_config

    def optimize_for_resource_usage(self) -> dict[str, Any]:
        """Get configuration optimized for low resource usage."""
        resource_config = self.optimized_config.copy()

        # Minimize resource usage
        resource_config["chunk_size"] = (
            RESOURCE_CHUNK_SIZE  # Larger chunks, less frequent processing
        )
        resource_config["buffer_size"] = DEFAULT_BUFFER_SIZE  # Smaller buffer
        resource_config["processing_interval"] = (
            RESOURCE_PROCESSING_INTERVAL  # Less frequent processing
        )
        resource_config["whisper_model_size"] = "tiny"  # Smallest model
        resource_config["device"] = "cpu"  # Force CPU to save GPU memory
        resource_config["compute_type"] = "int8"  # Most efficient compute type
        resource_config["max_audio_buffer_size"] = 5  # Smaller buffer

        logger.debug("Applied resource usage optimizations")
        return resource_config


class AdaptiveOptimizer:
    """Dynamically adapts configuration based on runtime performance."""

    def __init__(self, base_optimizer: PerformanceOptimizer):
        """Initialize adaptive optimizer."""
        self.base_optimizer = base_optimizer
        self.performance_history = []
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
        if len(self.performance_history) < 3:
            return False

        # Check if performance is consistently poor
        recent_latencies = [p["latency"] for p in self.performance_history[-3:]]
        recent_cpu = [p["cpu_usage"] for p in self.performance_history[-3:]]

        avg_latency = sum(recent_latencies) / len(recent_latencies)
        avg_cpu = sum(recent_cpu) / len(recent_cpu)

        # Adapt if latency is too high or CPU usage is too high
        return avg_latency > HIGH_LATENCY_THRESHOLD or avg_cpu > HIGH_CPU_THRESHOLD

    def adapt_configuration(self) -> dict[str, Any]:
        """Adapt configuration based on performance history."""
        if not self.should_adapt():
            return self.current_config

        self.adaptation_count += 1
        logger.debug(f"Adapting configuration (adaptation #{self.adaptation_count})")

        # Get recent performance metrics
        recent_metrics = self.performance_history[-3:]
        avg_latency = sum(p["latency"] for p in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(p["cpu_usage"] for p in recent_metrics) / len(recent_metrics)

        # Adapt based on performance issues
        if avg_latency > 5.0:
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
                0.5, self.current_config["max_silence_duration"] - 0.5
            )

        if avg_cpu > HIGH_CPU_THRESHOLD:
            # High CPU usage - reduce processing load
            logger.debug("High CPU usage detected, reducing processing load")
            self.current_config["chunk_size"] = min(
                RESOURCE_CHUNK_SIZE, self.current_config["chunk_size"] * 2
            )
            self.current_config["processing_interval"] = min(
                0.05, self.current_config["processing_interval"] * 2
            )

            # Switch to smaller model if using larger one
            if self.current_config["whisper_model_size"] == "large":
                self.current_config["whisper_model_size"] = "medium"
            elif self.current_config["whisper_model_size"] == "medium":
                self.current_config["whisper_model_size"] = "small"

        return self.current_config

    def get_current_config(self) -> dict[str, Any]:
        """Get current adapted configuration."""
        return self.current_config.copy()


# Global optimizer instance
_global_optimizer: PerformanceOptimizer | None = None
_global_optimizer_force_cpu: bool | None = None


def get_optimizer(
    use_cache: bool = True, force_refresh: bool = False, force_cpu: bool = False
) -> PerformanceOptimizer:
    """
    Get global performance optimizer instance.

    Args:
        use_cache: Whether to use cached optimization data
        force_refresh: Force recreation of optimizer (clears cache)
        force_cpu: Whether to force CPU-only mode
    """
    global _global_optimizer, _global_optimizer_force_cpu

    # Recreate optimizer if force_cpu setting has changed
    force_cpu_changed = (
        _global_optimizer_force_cpu is not None
        and _global_optimizer_force_cpu != force_cpu
    )

    if _global_optimizer is None or force_refresh or force_cpu_changed:
        _global_optimizer = PerformanceOptimizer(use_cache=use_cache, force_cpu=force_cpu)
        _global_optimizer_force_cpu = force_cpu
    return _global_optimizer


def get_optimized_config(
    optimization_target: str = "balanced", use_cache: bool = True, force_cpu: bool = False
) -> dict[str, Any]:
    """
    Get optimized configuration for specific target with caching.

    Args:
        optimization_target: "latency", "accuracy", "resource", or "balanced"
        use_cache: Whether to use cached configurations
        force_cpu: Whether to force CPU-only mode

    Returns:
        Optimized configuration dictionary
    """
    optimizer = get_optimizer(use_cache=use_cache, force_cpu=force_cpu)

    # Try to get from cache first
    if use_cache and optimizer.cache:
        cached_config = optimizer.cache.get_cached_config(optimization_target)
        if cached_config:
            return cached_config

    # Generate configuration
    if optimization_target == "latency":
        config = optimizer.optimize_for_latency()
    elif optimization_target == "accuracy":
        config = optimizer.optimize_for_accuracy()
    elif optimization_target == "resource":
        config = optimizer.optimize_for_resource_usage()
    else:  # balanced
        config = optimizer.optimized_config.copy()

    # Cache the result
    if use_cache and optimizer.cache:
        optimizer.cache.cache_config(optimization_target, config)

    return config


def clear_optimization_cache(cache_type: str = "all") -> None:
    """
    Clear optimization cache.

    Args:
        cache_type: Type of cache to clear ("system", "config", "performance", "all")
    """
    cache = get_optimization_cache()
    cache.clear_cache(cache_type)

    # Force refresh of global optimizer
    global _global_optimizer
    _global_optimizer = None


def get_cache_info() -> dict[str, Any]:
    """Get information about optimization cache."""
    cache = get_optimization_cache()
    return cache.get_cache_info()
