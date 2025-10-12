"""Integration tests for performance optimizations."""

import pytest
from local_ai.speech_to_text.service import SpeechToTextService
from local_ai.speech_to_text.optimization import PerformanceOptimizer, get_optimized_config


class TestOptimizationIntegration:
    """Test optimization integration with the speech-to-text service."""

    def test_optimizer_initialization(self) -> None:
        """Test that optimizer initializes correctly."""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.system_info is not None
        assert "cpu_count" in optimizer.system_info
        assert "platform" in optimizer.system_info
        assert optimizer.optimized_config is not None

    def test_optimized_service_initialization(self) -> None:
        """Test that service initializes with optimized configuration."""
        # Test different optimization targets
        targets = ["balanced", "latency", "accuracy", "resource"]
        
        for target in targets:
            service = SpeechToTextService(optimization_target=target)
            
            assert service._optimization_target == target
            assert service._optimized_config is not None
            assert service._optimizer is not None

    def test_optimized_config_generation(self) -> None:
        """Test that optimized configurations are generated correctly."""
        configs = {}
        
        for target in ["balanced", "latency", "accuracy", "resource"]:
            config = get_optimized_config(target)
            configs[target] = config
            
            # Verify required keys are present
            assert "chunk_size" in config
            assert "sample_rate" in config
            assert "whisper_model_size" in config
            assert "processing_interval" in config

        # Verify different targets produce different configurations
        latency_config = configs["latency"]
        accuracy_config = configs["accuracy"]
        resource_config = configs["resource"]
        
        # Latency optimization should have smaller chunk sizes and faster processing
        assert latency_config["chunk_size"] <= configs["balanced"]["chunk_size"]
        assert latency_config["processing_interval"] <= configs["balanced"]["processing_interval"]
        
        # Resource optimization should use smaller model
        assert resource_config["whisper_model_size"] in ["tiny", "small"]

    def test_component_optimization_integration(self) -> None:
        """Test that components use optimized configurations."""
        service = SpeechToTextService(optimization_target="latency")
        
        # Get component configurations
        audio_config = service._optimizer.get_optimized_audio_config()
        vad_config = service._optimizer.get_optimized_vad_config()
        transcriber_config = service._optimizer.get_optimized_transcriber_config()
        pipeline_config = service._optimizer.get_optimized_pipeline_config()
        
        # Verify configurations are dictionaries with expected keys
        assert isinstance(audio_config, dict)
        assert "sample_rate" in audio_config
        assert "chunk_size" in audio_config
        
        assert isinstance(vad_config, dict)
        assert "sample_rate" in vad_config
        assert "frame_duration" in vad_config
        
        assert isinstance(transcriber_config, dict)
        assert "model_size" in transcriber_config
        
        assert isinstance(pipeline_config, dict)
        assert "buffer_size" in pipeline_config
        assert "processing_interval" in pipeline_config

    @pytest.mark.integration
    def test_performance_monitoring_integration(self) -> None:
        """Test that performance monitoring is integrated."""
        # Create service with monitoring enabled
        service = SpeechToTextService(enable_monitoring=True)
        
        # Verify performance monitoring methods exist
        assert hasattr(service, 'get_performance_stats')
        assert hasattr(service, 'get_performance_report')
        assert hasattr(service, 'reset_performance_metrics')
        
        # Test performance stats (should work even without operations)
        stats = service.get_performance_stats()
        assert isinstance(stats, dict)
        assert "total_operations" in stats
        
        # Test performance report
        report = service.get_performance_report()
        assert isinstance(report, str)
        assert "Performance Report" in report
        
        # Test with monitoring disabled
        service_no_monitoring = SpeechToTextService(enable_monitoring=False)
        stats_disabled = service_no_monitoring.get_performance_stats()
        assert isinstance(stats_disabled, dict)
        assert "monitoring_disabled" in stats_disabled

    def test_optimization_target_effects(self) -> None:
        """Test that different optimization targets have measurable effects."""
        latency_service = SpeechToTextService(optimization_target="latency")
        accuracy_service = SpeechToTextService(optimization_target="accuracy")
        resource_service = SpeechToTextService(optimization_target="resource")
        
        latency_config = latency_service._optimized_config
        accuracy_config = accuracy_service._optimized_config
        resource_config = resource_service._optimized_config
        
        # Latency optimization should have faster processing intervals
        assert latency_config["processing_interval"] <= accuracy_config["processing_interval"]
        
        # Resource optimization should have larger chunk sizes (less frequent processing)
        assert resource_config["chunk_size"] >= latency_config["chunk_size"]
        
        # Accuracy optimization should allow longer speech segments
        assert accuracy_config["max_silence_duration"] >= latency_config["max_silence_duration"]

    def test_system_capability_detection(self) -> None:
        """Test that system capabilities are detected correctly."""
        optimizer = PerformanceOptimizer()
        capabilities = optimizer.system_info
        
        # Basic system info should be detected
        assert capabilities["cpu_count"] >= 1
        assert capabilities["platform"] in ["Linux", "Darwin", "Windows"]
        assert capabilities["memory_gb"] > 0
        
        # GPU detection should not fail (may be False)
        assert isinstance(capabilities["has_gpu"], bool)

    def test_adaptive_configuration(self) -> None:
        """Test adaptive configuration functionality."""
        from local_ai.speech_to_text.optimization import AdaptiveOptimizer
        
        base_optimizer = PerformanceOptimizer()
        adaptive = AdaptiveOptimizer(base_optimizer)
        
        # Initially should not adapt (no performance history)
        assert not adaptive.should_adapt()
        
        # Record some performance metrics
        adaptive.record_performance(latency=1.0, cpu_usage=50.0, memory_usage=1000.0)
        adaptive.record_performance(latency=1.2, cpu_usage=55.0, memory_usage=1100.0)
        adaptive.record_performance(latency=1.1, cpu_usage=52.0, memory_usage=1050.0)
        
        # Should still not adapt (performance is good)
        assert not adaptive.should_adapt()
        
        # Record poor performance
        adaptive.record_performance(latency=8.0, cpu_usage=90.0, memory_usage=2000.0)
        adaptive.record_performance(latency=7.5, cpu_usage=85.0, memory_usage=1900.0)
        adaptive.record_performance(latency=9.0, cpu_usage=95.0, memory_usage=2100.0)
        
        # Should now want to adapt
        assert adaptive.should_adapt()
        
        # Get adapted configuration
        adapted_config = adaptive.adapt_configuration()
        assert isinstance(adapted_config, dict)
        assert "chunk_size" in adapted_config
        assert "processing_interval" in adapted_config