"""Command-line interface for speech-to-text optimization management."""

import argparse
import sys

from .cache_utils import (
    clear_models_cache,
    get_cache_size,
    get_whisper_cache_dir,
)
from .optimization import (
    clear_optimization_cache,
    get_cache_info,
    get_optimized_config,
    get_optimizer,
)


def cmd_info(args: argparse.Namespace) -> None:
    """Show optimization and cache information."""

    # System capabilities
    try:
        optimizer = get_optimizer(use_cache=not args.no_cache)
        capabilities = optimizer.system_info

        if capabilities.get("has_gpu"):
            pass

    except Exception:
        pass

    # Cache information
    if not args.no_cache:
        try:
            cache_info = get_cache_info()

            for cache_type in ["system_cache", "config_cache", "performance_cache"]:
                info = cache_info[cache_type]
                "Valid" if info["valid"] else "Invalid/Expired"
                if info["exists"]:
                    pass
                else:
                    pass

        except Exception:
            pass

        # Show models cache info
        try:
            whisper_cache_dir = get_whisper_cache_dir()
            whisper_cache_size = get_cache_size(whisper_cache_dir)

            if whisper_cache_size > 0:
                # Count model files
                if whisper_cache_dir.exists():
                    len([f for f in whisper_cache_dir.rglob("*") if f.is_file()])
            else:
                pass

        except Exception:
            pass

    else:
        pass


def cmd_optimize(args: argparse.Namespace) -> None:
    """Generate and show optimized configuration."""

    try:
        config = get_optimized_config(
            optimization_target=args.target, use_cache=not args.no_cache
        )

        if args.json:
            pass
        else:
            for key, value in config.items():
                pass

    except Exception:
        sys.exit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run optimization benchmark."""

    targets = ["latency", "accuracy", "resource", "balanced"]
    results = {}

    for target in targets:
        try:
            import time

            start_time = time.time()

            config = get_optimized_config(
                optimization_target=target, use_cache=not args.no_cache
            )

            end_time = time.time()
            generation_time = end_time - start_time

            results[target] = {
                "generation_time_ms": generation_time * 1000,
                "config_size": len(str(config)),
                "success": True,
            }

        except Exception as e:
            results[target] = {"error": str(e), "success": False}

    # Summary
    successful = [r for r in results.values() if r.get("success")]
    if successful:
        sum(r["generation_time_ms"] for r in successful) / len(successful)

        if not args.no_cache:
            pass


def cmd_clear_cache(args: argparse.Namespace) -> None:
    """Clear optimization or models cache."""
    cache_type = args.type

    try:
        if cache_type == "models":
            success = clear_models_cache()
            if success:
                pass
            else:
                sys.exit(1)
        else:
            clear_optimization_cache(cache_type)

    except Exception:
        sys.exit(1)


def cmd_test_optimization(args: argparse.Namespace) -> None:
    """Test optimization system."""

    tests = [
        (
            "System Detection",
            lambda: get_optimizer(use_cache=not args.no_cache).system_info,
        ),
        (
            "Latency Config",
            lambda: get_optimized_config("latency", use_cache=not args.no_cache),
        ),
        (
            "Accuracy Config",
            lambda: get_optimized_config("accuracy", use_cache=not args.no_cache),
        ),
        (
            "Resource Config",
            lambda: get_optimized_config("resource", use_cache=not args.no_cache),
        ),
        (
            "Balanced Config",
            lambda: get_optimized_config("balanced", use_cache=not args.no_cache),
        ),
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                pass
        except Exception:
            pass

    if passed == total:
        sys.exit(0)
    else:
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

  # Clear only models cache
  python -m local_ai.speech_to_text.cli_optimization clear-cache --type models

  # Run benchmark without cache
  python -m local_ai.speech_to_text.cli_optimization benchmark --no-cache

  # Test optimization system
  python -m local_ai.speech_to_text.cli_optimization test
        """,
    )

    # Global options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache usage (force fresh optimization)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show optimization and cache information")

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Generate optimized configuration"
    )
    optimize_parser.add_argument(
        "--target",
        choices=["latency", "accuracy", "resource", "balanced"],
        default="balanced",
        help="Optimization target (default: balanced)",
    )
    optimize_parser.add_argument(
        "--json", action="store_true", help="Output configuration as JSON"
    )

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run optimization benchmark")

    # Clear cache command
    clear_parser = subparsers.add_parser("clear-cache", help="Clear optimization cache")
    clear_parser.add_argument(
        "--type",
        choices=["system", "config", "performance", "models", "all"],
        default="all",
        help="Type of cache to clear (default: all)",
    )

    # Test command
    subparsers.add_parser("test", help="Test optimization system")

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
