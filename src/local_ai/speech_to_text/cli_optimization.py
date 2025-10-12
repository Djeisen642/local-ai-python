"""Command-line interface for speech-to-text optimization management."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from .optimization import (
    get_optimizer, 
    get_optimized_config, 
    clear_optimization_cache, 
    get_cache_info
)
from .optimization_cache import get_optimization_cache
from .config import BYTES_PER_KB


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < BYTES_PER_KB:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= BYTES_PER_KB
    return f"{size_bytes:.1f} TB"


def cmd_info(args: argparse.Namespace) -> None:
    """Show optimization and cache information."""
    print("Speech-to-Text Optimization Information")
    print("=" * 50)
    
    # System capabilities
    try:
        optimizer = get_optimizer(use_cache=not args.no_cache)
        capabilities = optimizer.system_info
        
        print("\nSystem Capabilities:")
        print(f"  Platform: {capabilities.get('platform', 'Unknown')}")
        print(f"  Architecture: {capabilities.get('architecture', 'Unknown')}")
        print(f"  CPU Cores: {capabilities.get('cpu_count', 'Unknown')}")
        print(f"  Memory: {capabilities.get('memory_gb', 'Unknown'):.1f} GB")
        print(f"  GPU Available: {capabilities.get('has_gpu', False)}")
        if capabilities.get('has_gpu'):
            print(f"  GPU Memory: {capabilities.get('gpu_memory_gb', 0):.1f} GB")
    
    except Exception as e:
        print(f"\nError getting system info: {e}")
    
    # Cache information
    if not args.no_cache:
        print("\nCache Information:")
        try:
            cache_info = get_cache_info()
            print(f"  Cache Directory: {cache_info['cache_dir']}")
            
            for cache_type in ['system_cache', 'config_cache', 'performance_cache']:
                info = cache_info[cache_type]
                status = "Valid" if info['valid'] else "Invalid/Expired"
                if info['exists']:
                    print(f"  {cache_type.replace('_', ' ').title()}:")
                    print(f"    Status: {status}")
                    print(f"    Size: {format_size(info['size'])}")
                    print(f"    Age: {info['age_hours']:.1f} hours")
                else:
                    print(f"  {cache_type.replace('_', ' ').title()}: Not cached")
        
        except Exception as e:
            print(f"\nError getting cache info: {e}")
    else:
        print("\nCache: Disabled")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Generate and show optimized configuration."""
    print(f"Generating optimized configuration for: {args.target}")
    print("=" * 50)
    
    try:
        config = get_optimized_config(
            optimization_target=args.target,
            use_cache=not args.no_cache
        )
        
        if args.json:
            print(json.dumps(config, indent=2))
        else:
            print(f"\nOptimization Target: {args.target}")
            print("\nConfiguration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error generating configuration: {e}")
        sys.exit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run optimization benchmark."""
    print("Running optimization benchmark...")
    print("=" * 50)
    
    targets = ["latency", "accuracy", "resource", "balanced"]
    results = {}
    
    for target in targets:
        print(f"\nBenchmarking {target} optimization...")
        try:
            import time
            start_time = time.time()
            
            config = get_optimized_config(
                optimization_target=target,
                use_cache=not args.no_cache
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            results[target] = {
                "generation_time_ms": generation_time * 1000,
                "config_size": len(str(config)),
                "success": True
            }
            
            print(f"  Generated in {generation_time*1000:.2f}ms")
            
        except Exception as e:
            results[target] = {
                "error": str(e),
                "success": False
            }
            print(f"  Error: {e}")
    
    # Summary
    print("\nBenchmark Summary:")
    successful = [r for r in results.values() if r.get("success")]
    if successful:
        avg_time = sum(r["generation_time_ms"] for r in successful) / len(successful)
        print(f"  Average generation time: {avg_time:.2f}ms")
        print(f"  Successful optimizations: {len(successful)}/{len(targets)}")
        
        if not args.no_cache:
            print(f"  Cache usage: {'Enabled' if not args.no_cache else 'Disabled'}")


def cmd_clear_cache(args: argparse.Namespace) -> None:
    """Clear optimization cache."""
    cache_type = args.type
    
    print(f"Clearing {cache_type} cache...")
    
    try:
        clear_optimization_cache(cache_type)
        print(f"Successfully cleared {cache_type} cache")
    
    except Exception as e:
        print(f"Error clearing cache: {e}")
        sys.exit(1)


def cmd_test_optimization(args: argparse.Namespace) -> None:
    """Test optimization system."""
    print("Testing optimization system...")
    print("=" * 50)
    
    tests = [
        ("System Detection", lambda: get_optimizer(use_cache=not args.no_cache).system_info),
        ("Latency Config", lambda: get_optimized_config("latency", use_cache=not args.no_cache)),
        ("Accuracy Config", lambda: get_optimized_config("accuracy", use_cache=not args.no_cache)),
        ("Resource Config", lambda: get_optimized_config("resource", use_cache=not args.no_cache)),
        ("Balanced Config", lambda: get_optimized_config("balanced", use_cache=not args.no_cache)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name}: PASS")
                passed += 1
            else:
                print(f"✗ {test_name}: FAIL (empty result)")
        except Exception as e:
            print(f"✗ {test_name}: FAIL ({e})")
    
    print(f"\nTest Results: {passed}/{total} passed")
    
    if passed == total:
        print("All optimization tests passed!")
        sys.exit(0)
    else:
        print("Some optimization tests failed!")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Optimization Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show system and cache information
  python -m local_ai.speech_to_text.cli_optimization info
  
  # Generate latency-optimized configuration
  python -m local_ai.speech_to_text.cli_optimization optimize --target latency
  
  # Clear all cache and regenerate
  python -m local_ai.speech_to_text.cli_optimization clear-cache --type all
  
  # Run benchmark without cache
  python -m local_ai.speech_to_text.cli_optimization benchmark --no-cache
  
  # Test optimization system
  python -m local_ai.speech_to_text.cli_optimization test
        """
    )
    
    # Global options
    parser.add_argument(
        "--no-cache", 
        action="store_true", 
        help="Disable cache usage (force fresh optimization)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show optimization and cache information")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Generate optimized configuration")
    optimize_parser.add_argument(
        "--target", 
        choices=["latency", "accuracy", "resource", "balanced"],
        default="balanced",
        help="Optimization target (default: balanced)"
    )
    optimize_parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output configuration as JSON"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run optimization benchmark")
    
    # Clear cache command
    clear_parser = subparsers.add_parser("clear-cache", help="Clear optimization cache")
    clear_parser.add_argument(
        "--type",
        choices=["system", "config", "performance", "all"],
        default="all",
        help="Type of cache to clear (default: all)"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test optimization system")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "info":
        cmd_info(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "clear-cache":
        cmd_clear_cache(args)
    elif args.command == "test":
        cmd_test_optimization(args)


if __name__ == "__main__":
    main()