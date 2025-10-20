"""Unified cache utilities for the speech-to-text system."""

from pathlib import Path


def get_cache_root(cache_name: str | None = None) -> Path:
    """
    Get the root cache directory for the speech-to-text system.

    Args:
        cache_name: Optional subdirectory name within the cache root

    Returns:
        Path to the cache directory
    """
    cache_root = Path.home() / ".cache" / "local_ai" / "speech_to_text"

    if cache_name:
        cache_root = cache_root / cache_name

    # Ensure the directory exists
    cache_root.mkdir(parents=True, exist_ok=True)

    return cache_root


def get_optimization_cache_dir() -> Path:
    """Get the optimization cache directory."""
    return get_cache_root("optimization")


def get_models_cache_dir() -> Path:
    """Get the models cache directory."""
    return get_cache_root("models")


def get_whisper_cache_dir() -> Path:
    """Get the Whisper models cache directory."""
    return get_models_cache_dir() / "whisper"


def get_cache_size(cache_path: Path) -> int:
    """
    Get the total size of a cache directory in bytes.

    Args:
        cache_path: Path to the cache directory

    Returns:
        Total size in bytes
    """
    if not cache_path.exists():
        return 0

    total_size = 0
    try:
        for item in cache_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except (OSError, PermissionError):
        # Handle permission errors gracefully
        pass

    return total_size


def format_cache_size(size_bytes: int) -> str:
    """
    Format cache size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.1f} {units[unit_index]}"


def clear_models_cache() -> bool:
    """
    Clear the models cache directory.

    Returns:
        True if successful, False otherwise
    """
    import shutil

    try:
        models_cache_dir = get_models_cache_dir()
        if models_cache_dir.exists():
            shutil.rmtree(models_cache_dir)
            # Recreate the directory structure
            get_whisper_cache_dir()  # This will create the directory
            return True
        return True  # Nothing to clear
    except Exception:
        return False
