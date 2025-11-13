"""Additional tests to improve optimization.py coverage."""

from unittest.mock import MagicMock, patch

import pytest

from local_ai.speech_to_text.optimization import AdaptiveOptimizer, PerformanceOptimizer


@pytest.mark.unit
class TestOptimizationCoverage:
    """Test cases to improve coverage of optimization functionality."""

    def test_gpu_detection_with_torch_available(self) -> None:
        """Test GPU detection when torch is available and GPU is present."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = (
            8 * 1024**3
        )  # 8GB

        with patch.dict("sys.modules", {"torch": mock_torch}):
            optimizer = PerformanceOptimizer(use_cache=False)

        assert optimizer.system_info["has_gpu"] is True
        assert optimizer.system_info["gpu_memory_gb"] == 8.0

    def test_gpu_detection_with_torch_no_gpu(self) -> None:
        """Test GPU detection when torch is available but no GPU."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            optimizer = PerformanceOptimizer(use_cache=False)

        assert optimizer.system_info["has_gpu"] is False

    def test_gpu_detection_torch_import_error(self) -> None:
        """Test GPU detection when torch is not available."""
        # Patch the system detector function
        with patch(
            "local_ai.speech_to_text.performance_optimizer.detect_system_capabilities"
        ) as mock_detect:
            # Mock the return value to simulate torch import error
            mock_detect.return_value = {
                "cpu_count": 4,
                "platform": "Linux",
                "architecture": "x86_64",
                "has_gpu": False,
                "memory_gb": 8.0,
            }

            optimizer = PerformanceOptimizer(use_cache=False)

        assert optimizer.system_info["has_gpu"] is False

    def test_memory_detection_with_psutil(self) -> None:
        """Test memory detection when psutil is available."""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 16 * 1024**3  # 16GB

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            optimizer = PerformanceOptimizer(use_cache=False)

        assert optimizer.system_info["memory_gb"] == 16.0

    def test_memory_detection_psutil_import_error(self) -> None:
        """Test memory detection when psutil is not available."""
        # Patch the system detector function
        with patch(
            "local_ai.speech_to_text.performance_optimizer.detect_system_capabilities"
        ) as mock_detect:
            # Mock the return value to simulate psutil import error
            mock_detect.return_value = {
                "cpu_count": 4,
                "platform": "Linux",
                "architecture": "x86_64",
                "has_gpu": False,
                "memory_gb": 4,  # Default when psutil not available
            }

            optimizer = PerformanceOptimizer(use_cache=False)

        assert optimizer.system_info["memory_gb"] == 4  # Default

    def test_optimize_for_accuracy_with_gpu(self) -> None:
        """Test accuracy optimization with GPU available."""
        optimizer = PerformanceOptimizer(use_cache=False)

        # Mock GPU availability
        optimizer.system_info["has_gpu"] = True
        optimizer.system_info["gpu_memory_gb"] = 8.0

        config = optimizer.optimize_for_accuracy()

        # Should use distilled English-only medium model (default with OPT_USE_DISTILLED_MODELS=True)
        assert config["whisper_model_size"] == "distil-medium.en"

    def test_optimize_for_accuracy_with_large_gpu(self) -> None:
        """Test accuracy optimization with large GPU."""
        optimizer = PerformanceOptimizer(use_cache=False)

        # Mock large GPU
        optimizer.system_info["has_gpu"] = True
        optimizer.system_info["gpu_memory_gb"] = 12.0

        config = optimizer.optimize_for_accuracy()

        # Should use distilled large model (default with OPT_USE_DISTILLED_MODELS=True)
        assert config["whisper_model_size"] == "distil-large-v3"

    def test_platform_specific_optimizations(self) -> None:
        """Test platform-specific optimizations."""
        with patch("platform.system", return_value="Linux"):
            optimizer = PerformanceOptimizer(use_cache=False)

        # Linux should get more aggressive VAD
        vad_config = optimizer.get_optimized_vad_config()
        assert vad_config["aggressiveness"] >= 2  # Should be at least default

    def test_adaptive_optimizer_initialization(self) -> None:
        """Test AdaptiveOptimizer initialization."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        assert adaptive.base_optimizer == base_optimizer
        assert adaptive.performance_history == []
        assert adaptive.adaptation_count == 0

    def test_adaptive_optimizer_record_performance(self) -> None:
        """Test recording performance metrics."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        adaptive.record_performance(1.5, 60.0, 1000.0)

        assert len(adaptive.performance_history) == 1
        assert adaptive.performance_history[0]["latency"] == 1.5
        assert adaptive.performance_history[0]["cpu_usage"] == 60.0
        assert adaptive.performance_history[0]["memory_usage"] == 1000.0

    def test_adaptive_optimizer_history_limit(self) -> None:
        """Test that performance history is limited to 10 entries."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Add 15 entries
        for i in range(15):
            adaptive.record_performance(i, i * 10, i * 100)

        # Should only keep last 10
        assert len(adaptive.performance_history) == 10
        assert (
            adaptive.performance_history[0]["latency"] == 5
        )  # Entry 5 (0-indexed from 5-14)

    def test_adaptive_optimizer_should_not_adapt_insufficient_data(self) -> None:
        """Test that adaptation doesn't occur with insufficient data."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Add only 2 entries (need 3)
        adaptive.record_performance(1.0, 50.0, 1000.0)
        adaptive.record_performance(1.1, 55.0, 1100.0)

        assert not adaptive.should_adapt()

    def test_adaptive_optimizer_should_not_adapt_good_performance(self) -> None:
        """Test that adaptation doesn't occur with good performance."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Add 3 entries with good performance
        adaptive.record_performance(2.0, 50.0, 1000.0)
        adaptive.record_performance(2.1, 55.0, 1100.0)
        adaptive.record_performance(1.9, 52.0, 1050.0)

        assert not adaptive.should_adapt()

    def test_adaptive_optimizer_should_adapt_high_latency(self) -> None:
        """Test that adaptation occurs with high latency."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Add 3 entries with high latency
        adaptive.record_performance(8.0, 50.0, 1000.0)
        adaptive.record_performance(7.5, 55.0, 1100.0)
        adaptive.record_performance(9.0, 52.0, 1050.0)

        assert adaptive.should_adapt()

    def test_adaptive_optimizer_should_adapt_high_cpu(self) -> None:
        """Test that adaptation occurs with high CPU usage."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Add 3 entries with high CPU usage
        adaptive.record_performance(2.0, 90.0, 1000.0)
        adaptive.record_performance(2.1, 85.0, 1100.0)
        adaptive.record_performance(1.9, 95.0, 1050.0)

        assert adaptive.should_adapt()

    def test_adaptive_optimizer_adapt_for_high_latency(self) -> None:
        """Test adaptation configuration for high latency."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Record high latency
        adaptive.record_performance(8.0, 50.0, 1000.0)
        adaptive.record_performance(7.5, 55.0, 1100.0)
        adaptive.record_performance(9.0, 52.0, 1050.0)

        original_chunk_size = adaptive.current_config["chunk_size"]
        original_interval = adaptive.current_config["processing_interval"]

        adapted_config = adaptive.adapt_configuration()

        # Should reduce chunk size and processing interval for speed
        assert adapted_config["chunk_size"] < original_chunk_size
        assert adapted_config["processing_interval"] < original_interval
        assert adaptive.adaptation_count == 1

    def test_adaptive_optimizer_adapt_for_high_cpu(self) -> None:
        """Test adaptation configuration for high CPU usage."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Record high CPU usage
        adaptive.record_performance(2.0, 90.0, 1000.0)
        adaptive.record_performance(2.1, 85.0, 1100.0)
        adaptive.record_performance(1.9, 95.0, 1050.0)

        original_chunk_size = adaptive.current_config["chunk_size"]
        original_interval = adaptive.current_config["processing_interval"]

        adapted_config = adaptive.adapt_configuration()

        # Should increase chunk size and processing interval to reduce CPU load
        assert adapted_config["chunk_size"] > original_chunk_size
        assert adapted_config["processing_interval"] > original_interval

    def test_adaptive_optimizer_model_downgrade(self) -> None:
        """Test model downgrade for high CPU usage."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        # Set large model initially
        adaptive.current_config["whisper_model_size"] = "large"

        # Record high CPU usage
        adaptive.record_performance(2.0, 90.0, 1000.0)
        adaptive.record_performance(2.1, 85.0, 1100.0)
        adaptive.record_performance(1.9, 95.0, 1050.0)

        adapted_config = adaptive.adapt_configuration()

        # Should downgrade model to distilled medium with English-only (default with OPT_USE_DISTILLED_MODELS=True)
        assert adapted_config["whisper_model_size"] == "distil-medium.en"

    def test_adaptive_optimizer_get_current_config(self) -> None:
        """Test getting current adapted configuration."""
        base_optimizer = PerformanceOptimizer(use_cache=False)
        adaptive = AdaptiveOptimizer(base_optimizer)

        config = adaptive.get_current_config()

        # Should return a copy of current config
        assert isinstance(config, dict)
        assert config is not adaptive.current_config  # Should be a copy

    def test_optimizer_without_cache(self) -> None:
        """Test optimizer initialization without cache."""
        optimizer = PerformanceOptimizer(use_cache=False)

        assert optimizer.use_cache is False
        assert optimizer.cache is None

    def test_system_optimization_thresholds(self) -> None:
        """Test system optimization based on thresholds."""
        # Mock high-end system
        with patch("os.cpu_count", return_value=8):
            optimizer = PerformanceOptimizer(use_cache=False)
            optimizer.system_info["memory_gb"] = 16.0

            config = optimizer._generate_optimized_config()

            # Should get optimizations for high-end system
            assert config["chunk_size"] == 480  # Aligned with VAD frame (30ms at 16kHz)
            assert config["processing_interval"] == 0.005  # Faster processing

    def test_model_optimizations(self) -> None:
        """Test model optimization with distilled and English-only variants."""
        from local_ai.speech_to_text.optimization import apply_model_optimizations

        # Test with English-only enabled (default, distilled disabled by default)
        assert apply_model_optimizations("tiny", True, False) == "tiny.en"
        assert apply_model_optimizations("small", True, False) == "small.en"
        assert apply_model_optimizations("medium", True, False) == "medium.en"
        assert apply_model_optimizations("large", True, False) == "large"

        # Test with both distilled and English-only enabled (experimental)
        # When distilled is enabled, tiny/small/medium all upgrade to distil-medium
        assert (
            apply_model_optimizations("tiny", True, True) == "distil-medium.en"
        )  # Upgraded
        assert (
            apply_model_optimizations("small", True, True) == "distil-medium.en"
        )  # Upgraded
        assert (
            apply_model_optimizations("medium", True, True) == "distil-medium.en"
        )  # Distilled
        assert (
            apply_model_optimizations("large", True, True) == "distil-large-v3"
        )  # Distilled

        # Test with only distilled enabled (no English-only)
        assert apply_model_optimizations("tiny", False, True) == "distil-medium"
        assert apply_model_optimizations("small", False, True) == "distil-medium"
        assert apply_model_optimizations("medium", False, True) == "distil-medium"
        assert apply_model_optimizations("large", False, True) == "distil-large-v3"

        # Test with both disabled
        assert apply_model_optimizations("tiny", False, False) == "tiny"
        assert apply_model_optimizations("small", False, False) == "small"
        assert apply_model_optimizations("medium", False, False) == "medium"
        assert apply_model_optimizations("large", False, False) == "large"

    def test_default_config_uses_optimized_models(self) -> None:
        """Test that default configuration uses English-only models."""
        optimizer = PerformanceOptimizer(use_cache=False)
        config = optimizer.optimized_config

        model_size = config["whisper_model_size"]
        # Should use .en suffix for English-only optimization (unless large)
        is_optimized = model_size.endswith(".en") or model_size == "large"
        assert is_optimized, f"Expected English-only model, got {model_size}"
