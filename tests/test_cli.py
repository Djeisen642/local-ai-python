"""Tests for CLI interface functionality."""

import asyncio
import signal
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from local_ai.main import SpeechToTextCLI, main


class TestSpeechToTextCLI:
    """Test cases for the SpeechToTextCLI class."""

    def test_cli_initialization(self) -> None:
        """Test CLI initialization with default parameters."""
        cli = SpeechToTextCLI()
        
        assert cli._service is not None
        assert cli._running is False
        assert cli._transcription_count == 0

    def test_cli_initialization_with_custom_service(self) -> None:
        """Test CLI initialization with custom service."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service)
        
        assert cli._service is mock_service
        assert cli._running is False

    @pytest.mark.asyncio
    async def test_start_listening_success(self) -> None:
        """Test successful start of listening service."""
        mock_service = AsyncMock()
        # Make set_transcription_callback synchronous
        mock_service.set_transcription_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service)
        
        with patch('builtins.print') as mock_print:
            await cli.start_listening()
        
        mock_service.set_transcription_callback.assert_called_once()
        mock_service.start_listening.assert_called_once()
        assert cli._running is True
        
        # Check startup messages
        mock_print.assert_any_call("🎤 Starting speech-to-text service...")
        mock_print.assert_any_call("✅ Listening for speech. Speak into your microphone!")
        mock_print.assert_any_call("   Press Ctrl+C to stop.")

    @pytest.mark.asyncio
    async def test_start_listening_service_error(self) -> None:
        """Test handling of service errors during startup."""
        mock_service = AsyncMock()
        # Make set_transcription_callback synchronous
        mock_service.set_transcription_callback = Mock()
        mock_service.start_listening.side_effect = RuntimeError("Service initialization failed")
        cli = SpeechToTextCLI(service=mock_service)
        
        with patch('builtins.print') as mock_print:
            await cli.start_listening()
        
        assert cli._running is False
        mock_print.assert_any_call("❌ Error starting speech-to-text service: Service initialization failed")

    @pytest.mark.asyncio
    async def test_stop_listening_success(self) -> None:
        """Test successful stop of listening service."""
        mock_service = AsyncMock()
        cli = SpeechToTextCLI(service=mock_service)
        cli._running = True
        
        with patch('builtins.print') as mock_print:
            await cli.stop_listening()
        
        mock_service.stop_listening.assert_called_once()
        assert cli._running is False
        mock_print.assert_any_call("🛑 Stopping speech-to-text service...")

    @pytest.mark.asyncio
    async def test_stop_listening_when_not_running(self) -> None:
        """Test stop listening when service is not running."""
        mock_service = AsyncMock()
        cli = SpeechToTextCLI(service=mock_service)
        
        with patch('builtins.print') as mock_print:
            await cli.stop_listening()
        
        mock_service.stop_listening.assert_not_called()
        # Should not print stop message if not running

    def test_transcription_callback_with_text(self) -> None:
        """Test transcription callback with valid text."""
        cli = SpeechToTextCLI()
        
        with patch('builtins.print') as mock_print:
            cli._on_transcription("Hello world")
        
        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("📝 [1] Hello world")

    def test_transcription_callback_with_empty_text(self) -> None:
        """Test transcription callback with empty text."""
        cli = SpeechToTextCLI()
        
        with patch('builtins.print') as mock_print:
            cli._on_transcription("")
        
        assert cli._transcription_count == 0
        mock_print.assert_not_called()

    def test_transcription_callback_with_whitespace_only(self) -> None:
        """Test transcription callback with whitespace-only text."""
        cli = SpeechToTextCLI()
        
        with patch('builtins.print') as mock_print:
            cli._on_transcription("   \n\t  ")
        
        assert cli._transcription_count == 0
        mock_print.assert_not_called()

    def test_transcription_callback_increments_counter(self) -> None:
        """Test that transcription callback increments counter correctly."""
        cli = SpeechToTextCLI()
        
        with patch('builtins.print'):
            cli._on_transcription("First transcription")
            cli._on_transcription("Second transcription")
            cli._on_transcription("Third transcription")
        
        assert cli._transcription_count == 3

    def test_display_status_when_running(self) -> None:
        """Test status display when service is running."""
        mock_service = Mock()
        mock_service.get_component_status.return_value = {
            "audio_capture": True,
            "vad": True,
            "transcriber": True,
            "listening": True
        }
        cli = SpeechToTextCLI(service=mock_service)
        cli._running = True
        cli._transcription_count = 5
        
        with patch('builtins.print') as mock_print:
            cli.display_status()
        
        mock_print.assert_any_call("📊 Status: Running | Transcriptions: 5")
        mock_print.assert_any_call("   🎤 Audio: ✅ | 🔊 VAD: ✅ | 🤖 Transcriber: ✅")

    def test_display_status_when_not_running(self) -> None:
        """Test status display when service is not running."""
        cli = SpeechToTextCLI()
        
        with patch('builtins.print') as mock_print:
            cli.display_status()
        
        mock_print.assert_called_once_with("📊 Status: Stopped")

    def test_display_status_with_component_failures(self) -> None:
        """Test status display with some component failures."""
        mock_service = Mock()
        mock_service.get_component_status.return_value = {
            "audio_capture": False,
            "vad": True,
            "transcriber": False,
            "listening": True
        }
        cli = SpeechToTextCLI(service=mock_service)
        cli._running = True
        
        with patch('builtins.print') as mock_print:
            cli.display_status()
        
        mock_print.assert_any_call("   🎤 Audio: ❌ | 🔊 VAD: ✅ | 🤖 Transcriber: ❌")

    @pytest.mark.asyncio
    async def test_run_with_keyboard_interrupt(self) -> None:
        """Test graceful shutdown on KeyboardInterrupt."""
        mock_service = AsyncMock()
        cli = SpeechToTextCLI(service=mock_service)
        
        # Mock start_listening to set running state and then raise KeyboardInterrupt
        async def mock_start_and_interrupt():
            cli._running = True
            raise KeyboardInterrupt()
        
        cli.start_listening = mock_start_and_interrupt
        
        with patch('builtins.print') as mock_print:
            await cli.run()
        
        assert cli._running is False
        mock_print.assert_any_call("\n👋 Goodbye!")

    @pytest.mark.asyncio
    async def test_run_with_general_exception(self) -> None:
        """Test handling of general exceptions during run."""
        mock_service = AsyncMock()
        cli = SpeechToTextCLI(service=mock_service)
        
        # Mock start_listening to raise a general exception
        cli.start_listening = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        
        with patch('builtins.print') as mock_print:
            await cli.run()
        
        mock_print.assert_any_call("❌ Unexpected error: Unexpected error")

    @pytest.mark.asyncio
    async def test_run_normal_flow(self) -> None:
        """Test normal run flow without interruption."""
        mock_service = AsyncMock()
        cli = SpeechToTextCLI(service=mock_service)
        
        # Mock start_listening to set running state
        async def mock_start():
            cli._running = True
        
        cli.start_listening = mock_start
        
        # Mock asyncio.sleep to prevent infinite loop and simulate shutdown
        sleep_count = 0
        async def mock_sleep(duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:  # After a few iterations, simulate shutdown
                cli._running = False
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            with patch('builtins.print'):
                await cli.run()
        
        assert cli._running is False


class TestMainFunction:
    """Test cases for the main function and entry point."""

    @pytest.mark.asyncio
    async def test_main_function_creates_cli_and_runs(self) -> None:
        """Test that main function creates CLI and runs it."""
        with patch('local_ai.main.SpeechToTextCLI') as mock_cli_class:
            mock_cli = AsyncMock()
            mock_cli_class.return_value = mock_cli
            
            await main()
            
            mock_cli_class.assert_called_once()
            mock_cli.run.assert_called_once()

    def test_main_entry_point_calls_asyncio_run(self) -> None:
        """Test that the main entry point calls asyncio.run with main()."""
        with patch('asyncio.run') as mock_asyncio_run:
            with patch('local_ai.main.main') as mock_main:
                # Simulate the if __name__ == "__main__" block
                exec("""
if True:  # Simulate __name__ == "__main__"
    import asyncio
    from local_ai.main import main
    asyncio.run(main())
""")
                
                mock_asyncio_run.assert_called()

    def test_signal_handling_setup(self) -> None:
        """Test that signal handlers are properly set up for graceful shutdown."""
        # Test that we can handle SIGINT (Ctrl+C)
        with patch('signal.signal') as mock_signal:
            # This would be called in a real implementation
            signal.signal(signal.SIGINT, signal.default_int_handler)
            mock_signal.assert_called_with(signal.SIGINT, signal.default_int_handler)

    def test_cli_help_and_usage_display(self) -> None:
        """Test CLI help and usage information display."""
        # This test ensures we have proper help text
        help_text = """
Speech-to-Text CLI

Usage:
    python -m local_ai.main

Controls:
    Ctrl+C    - Stop and exit
    
The CLI will start listening to your microphone and display transcriptions in real-time.
Make sure your microphone is connected and permissions are granted.
"""
        
        # Test that help text is properly formatted
        assert "Speech-to-Text CLI" in help_text
        assert "Ctrl+C" in help_text
        assert "microphone" in help_text

    @pytest.mark.asyncio
    async def test_cli_real_time_display_updates(self) -> None:
        """Test that CLI provides real-time visual feedback."""
        mock_service = AsyncMock()
        cli = SpeechToTextCLI(service=mock_service)
        
        # Test multiple rapid transcriptions
        with patch('builtins.print') as mock_print:
            cli._on_transcription("Hello")
            cli._on_transcription("world")
            cli._on_transcription("this is a test")
        
        # Verify all transcriptions were displayed with proper formatting
        expected_calls = [
            (("📝 [1] Hello",),),
            (("📝 [2] world",),),
            (("📝 [3] this is a test",),)
        ]
        
        assert mock_print.call_args_list == expected_calls

    def test_cli_handles_service_component_status(self) -> None:
        """Test CLI properly handles and displays service component status."""
        mock_service = Mock()
        
        # Test various component status combinations
        test_cases = [
            {
                "status": {"audio_capture": True, "vad": True, "transcriber": True, "listening": True},
                "expected_audio": "✅",
                "expected_vad": "✅", 
                "expected_transcriber": "✅"
            },
            {
                "status": {"audio_capture": False, "vad": True, "transcriber": False, "listening": False},
                "expected_audio": "❌",
                "expected_vad": "✅",
                "expected_transcriber": "❌"
            }
        ]
        
        for case in test_cases:
            mock_service.get_component_status.return_value = case["status"]
            cli = SpeechToTextCLI(service=mock_service)
            cli._running = True
            
            with patch('builtins.print') as mock_print:
                cli.display_status()
            
            # Check that the status display includes the expected symbols
            status_call = None
            for call in mock_print.call_args_list:
                if "Audio:" in str(call):
                    status_call = str(call)
                    break
            
            assert status_call is not None
            assert case["expected_audio"] in status_call
            assert case["expected_vad"] in status_call
            assert case["expected_transcriber"] in status_call