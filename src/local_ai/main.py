"""Command-line interface for speech-to-text functionality."""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

from .speech_to_text.service import SpeechToTextService
from .speech_to_text.transcriber import WhisperTranscriber
from .speech_to_text.optimization_cache import get_optimization_cache


class SpeechToTextCLI:
    """Command-line interface for speech-to-text service."""

    def __init__(self, service: Optional[SpeechToTextService] = None, force_cpu: bool = False) -> None:
        """
        Initialize the CLI.
        
        Args:
            service: Optional SpeechToTextService instance. If None, creates a new one.
            force_cpu: Whether to force CPU-only mode
        """
        self._service = service or SpeechToTextService(force_cpu=force_cpu)
        self._running = False
        self._transcription_count = 0

    async def start_listening(self) -> None:
        """Start the speech-to-text listening service."""
        try:
            print("🎤 Starting speech-to-text service...")
            
            # Set up transcription callback
            self._service.set_transcription_callback(self._on_transcription)
            
            # Start the service
            await self._service.start_listening()
            
            self._running = True
            print("✅ Listening for speech. Speak into your microphone!")
            print("   Press Ctrl+C to stop.")
            
        except Exception as e:
            print(f"❌ Error starting speech-to-text service: {e}")
            self._running = False

    async def stop_listening(self) -> None:
        """Stop the speech-to-text listening service."""
        if not self._running:
            return
            
        print("🛑 Stopping speech-to-text service...")
        await self._service.stop_listening()
        self._running = False

    def _on_transcription(self, text: str) -> None:
        """
        Handle transcription results from the service.
        
        Args:
            text: Transcribed text
        """
        if not text or not text.strip():
            return
            
        self._transcription_count += 1
        print(f"📝 [{self._transcription_count}] {text}")

    def display_status(self) -> None:
        """Display current service status."""
        if not self._running:
            print("📊 Status: Stopped")
            return
            
        print(f"📊 Status: Running | Transcriptions: {self._transcription_count}")
        
        # Get component status
        status = self._service.get_component_status()
        audio_status = "✅" if status.get("audio_capture", False) else "❌"
        vad_status = "✅" if status.get("vad", False) else "❌"
        transcriber_status = "✅" if status.get("transcriber", False) else "❌"
        
        print(f"   🎤 Audio: {audio_status} | 🔊 VAD: {vad_status} | 🤖 Transcriber: {transcriber_status}")

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


async def main(force_cpu: bool = False) -> None:
    """Main entry point for the CLI application."""
    cli = SpeechToTextCLI(force_cpu=force_cpu)
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
        description="Speech-to-Text CLI - Convert real-time speech to text using local AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m local_ai.main                              # Start with default settings
  python -m local_ai.main --verbose                    # Enable verbose logging
  python -m local_ai.main --trace                      # Enable trace logging (most verbose)
  python -m local_ai.main --reset-model-cache          # Clear model cache and exit
  python -m local_ai.main --reset-optimization-cache   # Clear optimization cache and exit
  python -m local_ai.main -v --reset-model-cache       # Verbose mode with model cache reset
  python -m local_ai.main --force-cpu                  # Force CPU-only mode (disable GPU)

Controls:
  Ctrl+C    - Stop and exit gracefully
  
The CLI will start listening to your microphone and display transcriptions in real-time.
Make sure your microphone is connected and permissions are granted.
        """
    )
    
    parser.add_argument(
        '--reset-model-cache',
        action='store_true',
        help='Clear HuggingFace model cache and re-download models on next use'
    )
    
    parser.add_argument(
        '--reset-optimization-cache',
        action='store_true',
        help='Clear system optimization cache (capabilities, configs, performance)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging and debug information'
    )
    
    parser.add_argument(
        '--trace',
        action='store_true',
        help='Enable trace logging (most verbose, includes all debug info)'
    )
    
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU-only mode, disable GPU/CUDA acceleration'
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
    Reset the optimization cache by clearing system capabilities, configs, and performance data.
    
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
        - should_continue: True if execution should continue to main(), False if it should stop
    """
    # Import and setup trace logging
    from .speech_to_text.logging_utils import add_trace_level, TRACE_LEVEL
    add_trace_level()
    
    # Configure logging based on verbose/trace flags
    if args.trace:
        logging.basicConfig(
            level=TRACE_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Enable faster-whisper INFO logs in trace mode
        logging.getLogger("faster_whisper").setLevel(logging.INFO)
    elif args.verbose:
        logging.basicConfig(
            level='DEBUG',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Enable faster-whisper INFO logs in verbose mode
        logging.getLogger("faster_whisper").setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Suppress faster-whisper INFO logs in normal mode (only show WARNING and above)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    
    # Handle cache resets if requested
    cache_operations_performed = False
    
    if args.reset_model_cache:
        try:
            print("🔄 Clearing HuggingFace model cache...")
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
            print("🔄 Clearing system optimization cache...")
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


def cli_entry() -> None:
    """Synchronous entry point for CLI script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Final catch for any remaining KeyboardInterrupt


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
        asyncio.run(main(force_cpu=args.force_cpu))
        
    except KeyboardInterrupt:
        pass  # Graceful shutdown
    except SystemExit:
        # Re-raise SystemExit (from argparse help, etc.)
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_with_args()
