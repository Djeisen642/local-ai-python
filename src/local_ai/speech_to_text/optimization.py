"""
Performance optimizations for speech-to-text pipeline.

This module provides a public API for optimization functionality.
The implementation is split across multiple modules for better organization.
"""

from typing import Any

# Import from submodules
from .adaptive_optimizer import AdaptiveOptimizer
from .model_selector import apply_model_optimizations
from .performance_optimizer import PerformanceOptimizer
from .system_detector import detect_system_capabilities, validate_cuda_availability

# Public API exports
__all__ = [
    "PerformanceOptimizer",
    "AdaptiveOptimizer",
    "apply_model_optimizations",
    "detect_system_capabilities",
    "validate_cuda_availability",
    "get_optimizer",
    "get_optimized_config",
    "clear_optimization_cache",
    "get_cache_info",
]

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
    from .optimization_cache import get_optimization_cache

    cache = get_optimization_cache()
    cache.clear_cache(cache_type)

    # Force refresh of global optimizer
    global _global_optimizer
    _global_optimizer = None


def get_cache_info() -> dict[str, Any]:
    """Get information about optimization cache."""
    from .optimization_cache import get_optimization_cache

    cache = get_optimization_cache()
    return cache.get_cache_info()
