"""Command-line interface for speech-to-text functionality."""

import asyncio
import signal
from typing import Optional

from .speech_to_text.service import SpeechToTextService


class SpeechToTextCLI:
    """Command-line interface for speech-to-text service."""

    def __init__(self, service: Optional[SpeechToTextService] = None) -> None:
        """
        Initialize the CLI.
        
        Args:
            service: Optional SpeechToTextService instance. If None, creates a new one.
        """
        self._service = service or SpeechToTextService()
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
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except asyncio.CancelledError:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Ensure cleanup
            if self._running:
                await self.stop_listening()


async def main() -> None:
    """Main entry point for the CLI application."""
    cli = SpeechToTextCLI()
    try:
        await cli.run()
    except KeyboardInterrupt:
        pass  # Graceful shutdown already handled in cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Final catch for any remaining KeyboardInterrupt
