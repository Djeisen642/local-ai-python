"""Optimization caching system for speech-to-text pipeline."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

from .config import OPTIMIZATION_CACHE_SIZE

logger = logging.getLogger(__name__)


class OptimizationCache:
    """Manages caching of optimization configurations and system capabilities."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize optimization cache.
        
        Args:
            cache_dir: Directory to store cache files (default: ~/.cache/local_ai)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "local_ai" / "speech_to_text"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_cache_file = self.cache_dir / "system_capabilities.json"
        self.config_cache_file = self.cache_dir / "optimized_configs.json"
        self.performance_cache_file = self.cache_dir / "performance_history.json"
        
        # Cache validity duration (24 hours for system info, 1 hour for configs)
        self.system_cache_ttl = 24 * 60 * 60  # 24 hours
        self.config_cache_ttl = 60 * 60       # 1 hour
        self.performance_cache_ttl = 7 * 24 * 60 * 60  # 1 week
    
    def _get_system_fingerprint(self) -> str:
        """Generate a fingerprint of the current system for cache validation."""
        import platform
        import os
        
        # Create a hash of system characteristics that affect optimization
        system_data = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                system_data["gpu_name"] = torch.cuda.get_device_name(0)
                system_data["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass
        
        # Create hash
        system_str = json.dumps(system_data, sort_keys=True)
        return hashlib.md5(system_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path, ttl: int) -> bool:
        """Check if cache file is valid (exists and not expired)."""
        if not cache_file.exists():
            return False
        
        try:
            stat = cache_file.stat()
            age = time.time() - stat.st_mtime
            return age < ttl
        except OSError:
            return False
    
    def get_cached_system_capabilities(self) -> Optional[Dict[str, Any]]:
        """Get cached system capabilities if valid."""
        if not self._is_cache_valid(self.system_cache_file, self.system_cache_ttl):
            return None
        
        try:
            with open(self.system_cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Validate fingerprint
            current_fingerprint = self._get_system_fingerprint()
            if cached_data.get("fingerprint") != current_fingerprint:
                logger.info("System fingerprint changed, invalidating cache")
                return None
            
            logger.info("Using cached system capabilities")
            return cached_data.get("capabilities")
            
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to load system capabilities cache: {e}")
            return None
    
    def cache_system_capabilities(self, capabilities: Dict[str, Any]) -> None:
        """Cache system capabilities with fingerprint."""
        try:
            cache_data = {
                "fingerprint": self._get_system_fingerprint(),
                "capabilities": capabilities,
                "timestamp": time.time()
            }
            
            with open(self.system_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info("Cached system capabilities")
            
        except OSError as e:
            logger.warning(f"Failed to cache system capabilities: {e}")
    
    def get_cached_config(self, optimization_target: str) -> Optional[Dict[str, Any]]:
        """Get cached optimization config for target."""
        if not self._is_cache_valid(self.config_cache_file, self.config_cache_ttl):
            return None
        
        try:
            with open(self.config_cache_file, 'r') as f:
                cached_configs = json.load(f)
            
            # Check if we have a config for this target and system
            fingerprint = self._get_system_fingerprint()
            cache_key = f"{fingerprint}_{optimization_target}"
            
            if cache_key in cached_configs:
                logger.info(f"Using cached config for {optimization_target}")
                return cached_configs[cache_key]["config"]
            
            return None
            
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to load config cache: {e}")
            return None
    
    def cache_config(self, optimization_target: str, config: Dict[str, Any]) -> None:
        """Cache optimization config for target."""
        try:
            # Load existing cache
            cached_configs = {}
            if self.config_cache_file.exists():
                try:
                    with open(self.config_cache_file, 'r') as f:
                        cached_configs = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
            
            # Add new config
            fingerprint = self._get_system_fingerprint()
            cache_key = f"{fingerprint}_{optimization_target}"
            
            cached_configs[cache_key] = {
                "config": config,
                "timestamp": time.time()
            }
            
            # Clean old entries (keep only last N per system)
            system_entries = [(k, v) for k, v in cached_configs.items() if k.startswith(fingerprint)]
            if len(system_entries) > OPTIMIZATION_CACHE_SIZE:
                # Sort by timestamp and keep newest N
                system_entries.sort(key=lambda x: x[1]["timestamp"], reverse=True)
                for key, _ in system_entries[OPTIMIZATION_CACHE_SIZE:]:
                    del cached_configs[key]
            
            with open(self.config_cache_file, 'w') as f:
                json.dump(cached_configs, f, indent=2)
            
            logger.info(f"Cached config for {optimization_target}")
            
        except OSError as e:
            logger.warning(f"Failed to cache config: {e}")
    
    def get_cached_performance_history(self) -> Optional[Dict[str, Any]]:
        """Get cached performance history."""
        if not self._is_cache_valid(self.performance_cache_file, self.performance_cache_ttl):
            return None
        
        try:
            with open(self.performance_cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load performance cache: {e}")
            return None
    
    def cache_performance_history(self, history: Dict[str, Any]) -> None:
        """Cache performance history."""
        try:
            with open(self.performance_cache_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info("Cached performance history")
        except OSError as e:
            logger.warning(f"Failed to cache performance history: {e}")
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear cached data.
        
        Args:
            cache_type: Type of cache to clear ("system", "config", "performance", "all")
        """
        files_to_clear = []
        
        if cache_type in ("system", "all"):
            files_to_clear.append(self.system_cache_file)
        if cache_type in ("config", "all"):
            files_to_clear.append(self.config_cache_file)
        if cache_type in ("performance", "all"):
            files_to_clear.append(self.performance_cache_file)
        
        for cache_file in files_to_clear:
            try:
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Cleared cache: {cache_file.name}")
            except OSError as e:
                logger.warning(f"Failed to clear cache {cache_file.name}: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        info = {
            "cache_dir": str(self.cache_dir),
            "system_cache": {
                "exists": self.system_cache_file.exists(),
                "valid": self._is_cache_valid(self.system_cache_file, self.system_cache_ttl),
                "size": self.system_cache_file.stat().st_size if self.system_cache_file.exists() else 0,
                "age_hours": (time.time() - self.system_cache_file.stat().st_mtime) / 3600 if self.system_cache_file.exists() else 0
            },
            "config_cache": {
                "exists": self.config_cache_file.exists(),
                "valid": self._is_cache_valid(self.config_cache_file, self.config_cache_ttl),
                "size": self.config_cache_file.stat().st_size if self.config_cache_file.exists() else 0,
                "age_hours": (time.time() - self.config_cache_file.stat().st_mtime) / 3600 if self.config_cache_file.exists() else 0
            },
            "performance_cache": {
                "exists": self.performance_cache_file.exists(),
                "valid": self._is_cache_valid(self.performance_cache_file, self.performance_cache_ttl),
                "size": self.performance_cache_file.stat().st_size if self.performance_cache_file.exists() else 0,
                "age_hours": (time.time() - self.performance_cache_file.stat().st_mtime) / 3600 if self.performance_cache_file.exists() else 0
            }
        }
        
        return info


# Global cache instance
_global_cache: Optional[OptimizationCache] = None


def get_optimization_cache() -> OptimizationCache:
    """Get global optimization cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = OptimizationCache()
    return _global_cache