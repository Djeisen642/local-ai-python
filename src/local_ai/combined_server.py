"""Combined server that runs both speech-to-text and MCP server."""

import asyncio
import logging
import signal
import sys
from threading import Thread

from .speech_to_text.service import SpeechToTextService
from .task_management.config import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_MCP_HOST,
    DEFAULT_MCP_PORT,
)
from .task_management.database import TaskDatabase
from .task_management.mcp_server import mcp, set_task_manager
from .task_management.task_list_manager import TaskListManager

logger = logging.getLogger(__name__)


class CombinedServer:
    """Runs both speech-to-text service and MCP server together."""

    def __init__(
        self,
        mcp_transport: str = "sse",
        enable_audio_debugging: bool = False,
        audio_debug_dir: str | None = None,
    ) -> None:
        """
        Initialize combined server.

        Args:
            mcp_transport: MCP transport type ("stdio" or "sse")
            enable_audio_debugging: Whether to enable audio debugging
            audio_debug_dir: Optional directory for audio debug files
        """
        self.mcp_transport = mcp_transport
        self.enable_audio_debugging = enable_audio_debugging
        self.audio_debug_dir = audio_debug_dir
        self.stt_service: SpeechToTextService | None = None
        self.task_manager: TaskListManager | None = None
        self.mcp_thread: Thread | None = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize both services."""
        logger.info("Initializing combined server...")

        # Initialize database and task manager
        database = TaskDatabase(DEFAULT_DATABASE_PATH)
        await database.initialize()

        self.task_manager = TaskListManager(database)
        await self.task_manager.initialize()
        set_task_manager(self.task_manager)

        # Initialize task detection service
        from .task_management.llm_classifier import LLMClassifier
        from .task_management.task_detection_service import TaskDetectionService

        llm_classifier = LLMClassifier()
        task_detection_service = TaskDetectionService(
            llm_classifier=llm_classifier,
            task_manager=self.task_manager,
        )

        # Initialize speech-to-text service with task detection enabled
        self.stt_service = SpeechToTextService(
            enable_task_detection=True,
            task_detection_service=task_detection_service,
            enable_audio_debugging=self.enable_audio_debugging,
            audio_debug_dir=self.audio_debug_dir,
        )

        logger.info("Combined server initialized with task detection enabled")

    def start_mcp_server(self) -> None:
        """Start MCP server in a separate thread."""
        logger.info(f"Starting MCP server (transport={self.mcp_transport})...")

        if self.mcp_transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport="sse", host=DEFAULT_MCP_HOST, port=DEFAULT_MCP_PORT)

    async def start_stt_service(self) -> None:
        """Start speech-to-text service."""
        logger.info("Starting speech-to-text service...")

        def on_transcription(result):
            print(f"[STT] {result.text} ({round(result.confidence * 100)}%)")
            logger.info(
                f"🎤 Transcription received: '{result.text}' - triggering task detection"
            )

        self.stt_service.set_transcription_result_callback(on_transcription)
        await self.stt_service.start_listening()

        logger.info("Speech-to-text service started")

    async def run(self) -> None:
        """Run both services."""
        try:
            await self.initialize()

            # Start MCP server in background thread
            self.mcp_thread = Thread(target=self.start_mcp_server, daemon=True)
            self.mcp_thread.start()

            # Start speech-to-text service
            await self.start_stt_service()

            print("\n" + "=" * 70)
            print("🚀 Combined Server Running")
            print("=" * 70)
            print("🎤 Speech-to-Text: Listening for speech...")
            if self.mcp_transport == "sse":
                print(f"🔗 MCP Server: http://{DEFAULT_MCP_HOST}:{DEFAULT_MCP_PORT}/sse")
            else:
                print("📡 MCP Server: stdio transport")
            print("=" * 70)
            print("Press Ctrl+C to stop\n")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown both services."""
        logger.info("Shutting down combined server...")

        if self.stt_service:
            await self.stt_service.stop_listening()

        if self.task_manager:
            await self.task_manager.shutdown()

        logger.info("Combined server shutdown complete")


def cli_entry() -> None:
    """CLI entry point for combined server."""
    import argparse

    parser = argparse.ArgumentParser(description="Combined Speech-to-Text and MCP Server")
    parser.add_argument(
        "--mcp-transport",
        choices=["stdio", "sse"],
        default="sse",
        help="MCP transport type (default: sse)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
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

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n👋 Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run combined server
    server = CombinedServer(
        mcp_transport=args.mcp_transport,
        enable_audio_debugging=args.debug_audio,
        audio_debug_dir=args.debug_audio_dir,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    cli_entry()
