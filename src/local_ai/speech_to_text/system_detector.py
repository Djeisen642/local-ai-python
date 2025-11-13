"""System capability detection for optimization."""

from typing import Any

from .config import BYTES_PER_KB, OPT_DEFAULT_GPU_MEMORY_GB, OPT_DEFAULT_MEMORY_GB
from .logging_utils import get_logger

logger = get_logger(__name__)


def detect_system_capabilities(force_cpu: bool = False) -> dict[str, Any]:
    """
    Detect system capabilities for optimization.

    Args:
        force_cpu: Whether to force CPU-only mode (disable GPU/CUDA)

    Returns:
        Dictionary containing system capabilities
    """
    import os
    import platform

    capabilities = {
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "has_gpu": False,
        "memory_gb": OPT_DEFAULT_MEMORY_GB,
    }

    # Check if GPU should be forced off
    if force_cpu:
        logger.debug("GPU detection disabled - force CPU mode requested")
        capabilities["has_gpu"] = False
    else:
        # Try to detect GPU capabilities
        capabilities["has_gpu"] = validate_cuda_availability()
        if capabilities["has_gpu"]:
            # Try to get GPU memory info
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                    capabilities["gpu_memory_gb"] = gpu_memory_bytes / (BYTES_PER_KB**3)
                    logger.debug(
                        f"GPU detected: {capabilities['gpu_memory_gb']:.1f}GB memory"
                    )
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")
                capabilities["gpu_memory_gb"] = OPT_DEFAULT_GPU_MEMORY_GB

    # Try to detect memory
    try:
        import psutil

        capabilities["memory_gb"] = psutil.virtual_memory().total / (BYTES_PER_KB**3)
    except ImportError:
        pass

    logger.debug(f"Detected system capabilities: {capabilities}")
    return capabilities


def validate_cuda_availability() -> bool:
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
