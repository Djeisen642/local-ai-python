"""Command-line interface for speech-to-text functionality."""

import argparse
import asyncio
import logging
import sys

from .speech_to_text.models import TranscriptionResult
from .speech_to_text.optimization_cache import get_optimization_cache
from .speech_to_text.service import SpeechToTextService
from .speech_to_text.transcriber import WhisperTranscriber


class SpeechToTextCLI:
    """Command-line interface for speech-to-text service."""

    def __init__(
        self,
        service: SpeechToTextService | None = None,
        force_cpu: bool = False,
        show_confidence_percentage: bool = True,
        enable_audio_debugging: bool = False,
        audio_debug_dir: str | None = None,
    ) -> None:
        """
        Initialize the CLI.

        Args:
            service: Optional SpeechToTextService instance. If None, creates a new one.
            force_cpu: Whether to force CPU-only mode
            show_confidence_percentage: Whether to show confidence percentages in output
            enable_audio_debugging: Whether to enable audio debugging
            audio_debug_dir: Optional directory for audio debug files
        """
        self._service = service or SpeechToTextService(
            force_cpu=force_cpu,
            enable_audio_debugging=enable_audio_debugging,
            audio_debug_dir=audio_debug_dir,
        )
        self._running = False
        self._transcription_count = 0
        self._show_confidence_percentage = show_confidence_percentage
        self._latest_transcription_result: TranscriptionResult | None = None

    async def start_listening(self) -> None:
        """Start the speech-to-text listening service."""
        try:
            print("🎤 Starting speech-to-text service...")

            # Set up transcription callback (confidence-based to avoid double printing)
            self._service.set_transcription_result_callback(self._on_transcription_result)

            # Start the service
            await self._service.start_listening()

            self._running = True
            print("✅ Listening for speech. Speak into your microphone!")
            print("   Press Ctrl+C to stop.")

        except Exception as e:
            self._running = False
            print(f"❌ Error starting speech-to-text service: {e}")

    async def stop_listening(self) -> None:
        """Stop the speech-to-text listening service."""
        if not self._running:
            return

        print("🛑 Stopping speech-to-text service...")
        await self._service.stop_listening()
        self._running = False

    def _on_transcription_result(self, result: TranscriptionResult) -> None:
        """
        Handle transcription results with confidence information from the service.

        Args:
            result: TranscriptionResult with confidence information
        """
        if not result.text or not result.text.strip():
            return

        self._transcription_count += 1

        # Store the latest result for downstream systems
        self._latest_transcription_result = result

        # Format the main transcription display with confidence percentage
        if self._show_confidence_percentage:
            confidence_percent = round(result.confidence * 100)
            print(f"[{self._transcription_count}] {result.text} ({confidence_percent}%)")
        else:
            print(f"[{self._transcription_count}] {result.text}")

    async def run(self) -> None:
        """
        Main CLI run loop.

        Handles startup, main loop, and graceful shutdown.
        """
        try:
            # Start listening
            await self.start_listening()

            # Main loop - keep running until interrupted
            while self._running:
                await asyncio.sleep(0.1)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Ensure cleanup
            if self._running:
                await self.stop_listening()


async def main(
    force_cpu: bool = False,
    show_confidence_percentage: bool = True,
    enable_audio_debugging: bool = False,
    audio_debug_dir: str | None = None,
) -> None:
    """Main entry point for the CLI application."""
    cli = SpeechToTextCLI(
        force_cpu=force_cpu,
        show_confidence_percentage=show_confidence_percentage,
        enable_audio_debugging=enable_audio_debugging,
        audio_debug_dir=audio_debug_dir,
    )
    try:
        await cli.run()
    except KeyboardInterrupt:
        pass  # Graceful shutdown already handled in cli.run()


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Text CLI - Convert speech to text using local AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m local_ai.main                              # Start with defaults
  python -m local_ai.main --no-confidence              # Hide confidence %
  python -m local_ai.main --verbose                    # Enable verbose logging
  python -m local_ai.main --trace                      # Enable trace logging
  python -m local_ai.main --reset-model-cache          # Clear model cache
  python -m local_ai.main --reset-optimization-cache   # Clear optimization cache
  python -m local_ai.main -v --reset-model-cache       # Verbose + cache reset
  python -m local_ai.main --force-cpu                  # Force CPU-only mode (disable GPU)
  python -m local_ai.main --debug-audio                # Enable audio debugging
  python -m local_ai.main --debug-audio --debug-audio-dir /tmp/audio  # Custom debug directory

Controls:
  Ctrl+C    - Stop and exit gracefully

The CLI will start listening to your microphone and display transcriptions in real-time.
Make sure your microphone is connected and permissions are granted.
        """,
    )

    parser.add_argument(
        "--reset-model-cache",
        action="store_true",
        help="Clear HuggingFace model cache and re-download models on next use",
    )

    parser.add_argument(
        "--reset-optimization-cache",
        action="store_true",
        help="Clear system optimization cache (capabilities, configs, performance)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging and debug information",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable trace logging (most verbose, includes all debug info)",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only mode, disable GPU/CUDA acceleration",
    )

    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Hide confidence percentages in transcription output",
    )

    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Enable audio debugging (save processed audio to WAV files)",
    )

    parser.add_argument(
        "--debug-audio-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Custom output directory for audio debug files (default: ~/.cache/local_ai/audio_debug)",
    )

    return parser


def reset_model_cache() -> bool:
    """
    Reset the model cache by clearing downloaded models.

    Returns:
        True if cache was cleared successfully, False otherwise
    """
    try:
        transcriber = WhisperTranscriber()
        return transcriber.clear_model_cache()
    except Exception as e:
        logging.error(f"Error during model cache reset: {e}")
        return False


def reset_optimization_cache() -> bool:
    """
    Reset optimization cache by clearing system capabilities and performance data.

    Returns:
        True if cache was cleared successfully, False otherwise
    """
    try:
        cache = get_optimization_cache()
        cache.clear_cache("all")
        return True
    except Exception as e:
        logging.error(f"Error during optimization cache reset: {e}")
        return False


def handle_arguments(args: argparse.Namespace) -> tuple[bool, bool]:
    """
    Handle parsed command-line arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        Tuple of (success, should_continue):
        - success: True if all operations succeeded, False if any failed
        - should_continue: True if execution should continue, False if it should stop
    """
    # Import and setup trace logging
    from .speech_to_text.logging_utils import TRACE_LEVEL, add_trace_level

    add_trace_level()

    # Configure logging based on verbose/trace flags
    if args.trace:
        logging.basicConfig(
            level=TRACE_LEVEL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        # Enable faster-whisper INFO logs in trace mode
        logging.getLogger("faster_whisper").setLevel(logging.INFO)
    elif args.verbose:
        logging.basicConfig(
            level="DEBUG", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Enable faster-whisper INFO logs in verbose mode
        logging.getLogger("faster_whisper").setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level="INFO", format="%(asctime)s - %(levelname)s - %(message)s"
        )
        # Suppress faster-whisper INFO logs in normal mode (only show WARNING and above)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    # Handle cache resets if requested
    cache_operations_performed = False

    if args.reset_model_cache:
        try:
            success = reset_model_cache()
            if success:
                print("✅ Model cache cleared successfully.")
            else:
                print("❌ Failed to clear model cache.")
                return False, False
            cache_operations_performed = True
        except Exception as e:
            print(f"❌ Error clearing model cache: {e}")
            return False, False

    if args.reset_optimization_cache:
        try:
            success = reset_optimization_cache()
            if success:
                print("✅ Optimization cache cleared successfully.")
            else:
                print("❌ Failed to clear optimization cache.")
                return False, False
            cache_operations_performed = True
        except Exception as e:
            print(f"❌ Error clearing optimization cache: {e}")
            return False, False

    # If cache operations were performed successfully, don't continue to main execution
    if cache_operations_performed:
        return True, False

    return True, True


def cli_entry_with_args() -> None:
    """CLI entry point with argument parsing."""
    parser = create_argument_parser()

    try:
        args = parser.parse_args()

        # Handle arguments and check if we should continue
        success, should_continue = handle_arguments(args)

        if not success:
            sys.exit(1)

        if not should_continue:
            sys.exit(0)

        # If we get here, proceed with normal execution
        show_confidence = (
            not args.no_confidence
        )  # Invert the flag since --no-confidence hides it
        asyncio.run(
            main(
                force_cpu=args.force_cpu,
                show_confidence_percentage=show_confidence,
                enable_audio_debugging=args.debug_audio,
                audio_debug_dir=args.debug_audio_dir,
            )
        )

    except KeyboardInterrupt:
        pass  # Graceful shutdown
    except SystemExit:
        # Re-raise SystemExit (from argparse help, etc.)
        raise
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_with_args()
