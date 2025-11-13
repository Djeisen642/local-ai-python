"""Main performance optimizer class."""

from typing import Any

from .config import (
    ACCURACY_BUFFER_SIZE,
    ACCURACY_MAX_SILENCE_DURATION,
    ACCURACY_MIN_SPEECH_DURATION,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_AUDIO_BUFFER_SIZE,
    DEFAULT_MAX_SILENCE_DURATION,
    DEFAULT_MIN_SPEECH_DURATION,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TRANSCRIPTION_TIMEOUT,
    HIGH_GPU_MEMORY_GB,
    HIGH_MEMORY_BUFFER_SIZE,
    HIGH_MEMORY_GB,
    LATENCY_BUFFER_SIZE,
    LATENCY_CHUNK_SIZE,
    LATENCY_MAX_SILENCE_DURATION,
    LATENCY_MIN_SPEECH_DURATION,
    LATENCY_VAD_FRAME_DURATION,
    OPT_COMPUTE_TYPE_FLOAT16,
    OPT_COMPUTE_TYPE_INT8,
    OPT_CPU_CORES_FAST,
    OPT_CPU_CORES_MANY,
    OPT_DEFAULT_GPU_MEMORY_GB,
    OPT_DEVICE_CPU,
    OPT_DEVICE_CUDA,
    OPT_MAX_AUDIO_BUFFER_HIGH_MEM,
    OPT_MAX_AUDIO_BUFFER_RESOURCE,
    OPT_MAX_CONCURRENT_TRANSCRIPTIONS,
    OPT_MODEL_SIZE_DEFAULT,
    OPT_MODEL_SIZE_LARGE,
    OPT_MODEL_SIZE_MEDIUM,
    OPT_MODEL_SIZE_SMALL,
    OPT_MODEL_SIZE_TINY,
    OPT_PLATFORM_LINUX,
    OPT_PROCESSING_INTERVAL_DEFAULT,
    OPT_PROCESSING_INTERVAL_FAST_CPU,
    OPT_PROCESSING_INTERVAL_LATENCY,
    OPT_USE_DISTILLED_MODELS,
    OPT_USE_ENGLISH_ONLY_MODEL,
    OPT_VAD_AGGRESSIVENESS_ACCURACY_REDUCE,
    OPT_VAD_AGGRESSIVENESS_LINUX_BOOST,
    RESOURCE_CHUNK_SIZE,
    RESOURCE_PROCESSING_INTERVAL,
    ULTRA_GPU_MEMORY_GB,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION,
)
from .logging_utils import get_logger
from .model_selector import apply_model_optimizations
from .optimization_cache import get_optimization_cache
from .system_detector import detect_system_capabilities

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
        self.system_info = detect_system_capabilities(force_cpu=force_cpu)
        self.optimized_config = self._generate_optimized_config()

    def _generate_optimized_config(self) -> dict[str, Any]:
        """Generate optimized configuration based on system capabilities."""
        # Try to get from cache first
        if self.use_cache and self.cache:
            cached_config = self.cache.get_cached_config("balanced")
            if cached_config:
                return cached_config

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
            "whisper_model_size": apply_model_optimizations(
                OPT_MODEL_SIZE_DEFAULT,
                OPT_USE_ENGLISH_ONLY_MODEL,
                OPT_USE_DISTILLED_MODELS,
            ),
            "compute_type": OPT_COMPUTE_TYPE_INT8,
            "device": OPT_DEVICE_CPU,
            "max_audio_buffer_size": DEFAULT_MAX_AUDIO_BUFFER_SIZE,
            "transcription_timeout": DEFAULT_TRANSCRIPTION_TIMEOUT,
            # Pipeline optimizations
            "processing_interval": OPT_PROCESSING_INTERVAL_DEFAULT,
            "max_concurrent_transcriptions": OPT_MAX_CONCURRENT_TRANSCRIPTIONS,
        }

        # Optimize based on system capabilities
        if self.system_info["cpu_count"] >= OPT_CPU_CORES_FAST:
            config["chunk_size"] = LATENCY_CHUNK_SIZE
            config["processing_interval"] = OPT_PROCESSING_INTERVAL_FAST_CPU

        if self.system_info["memory_gb"] >= HIGH_MEMORY_GB:
            config["buffer_size"] = HIGH_MEMORY_BUFFER_SIZE
            config["max_audio_buffer_size"] = OPT_MAX_AUDIO_BUFFER_HIGH_MEM

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
            config["device"] = OPT_DEVICE_CPU
            config["compute_type"] = OPT_COMPUTE_TYPE_INT8
        else:
            logger.debug("Using GPU acceleration for Whisper transcription")
            config["device"] = OPT_DEVICE_CUDA
            config["compute_type"] = OPT_COMPUTE_TYPE_FLOAT16

            # Adjust model size based on GPU memory
            gpu_memory = self.system_info.get("gpu_memory_gb", OPT_DEFAULT_GPU_MEMORY_GB)
            if gpu_memory >= ULTRA_GPU_MEMORY_GB:
                config["whisper_model_size"] = apply_model_optimizations(
                    OPT_MODEL_SIZE_LARGE,
                    OPT_USE_ENGLISH_ONLY_MODEL,
                    OPT_USE_DISTILLED_MODELS,
                )
            elif gpu_memory >= HIGH_GPU_MEMORY_GB:
                config["whisper_model_size"] = apply_model_optimizations(
                    OPT_MODEL_SIZE_MEDIUM,
                    OPT_USE_ENGLISH_ONLY_MODEL,
                    OPT_USE_DISTILLED_MODELS,
                )
            else:
                config["whisper_model_size"] = apply_model_optimizations(
                    OPT_MODEL_SIZE_SMALL,
                    OPT_USE_ENGLISH_ONLY_MODEL,
                    OPT_USE_DISTILLED_MODELS,
                )

        # Optimize CPU performance based on system capabilities
        if self.system_info["cpu_count"] >= OPT_CPU_CORES_MANY:
            config["whisper_model_size"] = apply_model_optimizations(
                OPT_MODEL_SIZE_SMALL,
                OPT_USE_ENGLISH_ONLY_MODEL,
                OPT_USE_DISTILLED_MODELS,
            )
        elif self.system_info["cpu_count"] >= OPT_CPU_CORES_FAST:
            config["whisper_model_size"] = apply_model_optimizations(
                OPT_MODEL_SIZE_TINY,
                OPT_USE_ENGLISH_ONLY_MODEL,
                OPT_USE_DISTILLED_MODELS,
            )

        # Platform-specific optimizations
        if self.system_info["platform"] == OPT_PLATFORM_LINUX:
            config["vad_aggressiveness"] = min(
                OPT_VAD_AGGRESSIVENESS_LINUX_BOOST + 2,
                VAD_AGGRESSIVENESS + OPT_VAD_AGGRESSIVENESS_LINUX_BOOST,
            )

        logger.debug(f"Generated optimized config: {config}")

        # Cache the result
        if self.use_cache and self.cache:
            self.cache.cache_config("balanced", config)

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
        latency_config["processing_interval"] = OPT_PROCESSING_INTERVAL_LATENCY
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
            OPT_VAD_AGGRESSIVENESS_ACCURACY_REDUCE,
            VAD_AGGRESSIVENESS - OPT_VAD_AGGRESSIVENESS_ACCURACY_REDUCE,
        )

        # Use better model if system supports it
        gpu_memory = self.system_info.get("gpu_memory_gb", OPT_DEFAULT_GPU_MEMORY_GB)
        if gpu_memory >= HIGH_GPU_MEMORY_GB:
            accuracy_config["whisper_model_size"] = apply_model_optimizations(
                OPT_MODEL_SIZE_MEDIUM,
                OPT_USE_ENGLISH_ONLY_MODEL,
                OPT_USE_DISTILLED_MODELS,
            )
        if gpu_memory >= ULTRA_GPU_MEMORY_GB:
            accuracy_config["whisper_model_size"] = apply_model_optimizations(
                OPT_MODEL_SIZE_LARGE,
                OPT_USE_ENGLISH_ONLY_MODEL,
                OPT_USE_DISTILLED_MODELS,
            )

        logger.debug("Applied accuracy optimizations")
        return accuracy_config

    def optimize_for_resource_usage(self) -> dict[str, Any]:
        """Get configuration optimized for low resource usage."""
        resource_config = self.optimized_config.copy()

        # Minimize resource usage
        resource_config["chunk_size"] = RESOURCE_CHUNK_SIZE
        resource_config["buffer_size"] = DEFAULT_BUFFER_SIZE
        resource_config["processing_interval"] = RESOURCE_PROCESSING_INTERVAL
        resource_config["whisper_model_size"] = apply_model_optimizations(
            OPT_MODEL_SIZE_TINY,
            OPT_USE_ENGLISH_ONLY_MODEL,
            OPT_USE_DISTILLED_MODELS,
        )
        resource_config["device"] = OPT_DEVICE_CPU
        resource_config["compute_type"] = OPT_COMPUTE_TYPE_INT8
        resource_config["max_audio_buffer_size"] = OPT_MAX_AUDIO_BUFFER_RESOURCE

        logger.debug("Applied resource usage optimizations")
        return resource_config
