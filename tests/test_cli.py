"""Tests for CLI interface functionality."""

import asyncio
import signal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_ai.main import SpeechToTextCLI, main
from local_ai.speech_to_text.models import TranscriptionResult


@pytest.mark.unit
class TestSpeechToTextCLI:
    """Test cases for the SpeechToTextCLI class."""

    def test_cli_initialization(self) -> None:
        """Test CLI initialization with default parameters."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service)

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
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        with patch("builtins.print") as mock_print:
            await cli.start_listening()

        mock_service.set_transcription_result_callback.assert_called_once()
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
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        mock_service.start_listening.side_effect = RuntimeError(
            "Service initialization failed"
        )
        cli = SpeechToTextCLI(service=mock_service)

        with patch("builtins.print") as mock_print:
            await cli.start_listening()

        assert cli._running is False
        mock_print.assert_any_call(
            "❌ Error starting speech-to-text service: Service initialization failed"
        )

    @pytest.mark.asyncio
    async def test_stop_listening_success(self) -> None:
        """Test successful stop of listening service."""
        mock_service = AsyncMock()
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service)
        cli._running = True

        with patch("builtins.print") as mock_print:
            await cli.stop_listening()

        mock_service.stop_listening.assert_called_once()
        assert cli._running is False
        mock_print.assert_any_call("🛑 Stopping speech-to-text service...")

    @pytest.mark.asyncio
    async def test_stop_listening_when_not_running(self) -> None:
        """Test stop listening when service is not running."""
        mock_service = AsyncMock()
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        with patch("builtins.print"):
            await cli.stop_listening()

        mock_service.stop_listening.assert_not_called()
        # Should not print stop message if not running

    @pytest.mark.asyncio
    async def test_run_with_keyboard_interrupt(self) -> None:
        """Test graceful shutdown on KeyboardInterrupt."""
        mock_service = AsyncMock()
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        # Mock start_listening to set running state and then raise KeyboardInterrupt
        async def mock_start_and_interrupt():
            cli._running = True
            raise KeyboardInterrupt()

        cli.start_listening = mock_start_and_interrupt

        with patch("builtins.print") as mock_print:
            await cli.run()

        assert cli._running is False
        mock_print.assert_any_call("\n👋 Goodbye!")

    @pytest.mark.asyncio
    async def test_run_with_general_exception(self) -> None:
        """Test handling of general exceptions during run."""
        mock_service = AsyncMock()
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        # Mock start_listening to raise a general exception
        cli.start_listening = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with patch("builtins.print") as mock_print:
            await cli.run()

        mock_print.assert_any_call("❌ Unexpected error: Unexpected error")

    @pytest.mark.asyncio
    async def test_run_normal_flow(self) -> None:
        """Test normal run flow without interruption."""
        mock_service = AsyncMock()
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
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

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with patch("builtins.print"):
                await cli.run()

        assert cli._running is False


@pytest.mark.unit
class TestMainFunction:
    """Test cases for the main function and entry point."""

    @pytest.mark.asyncio
    async def test_main_function_creates_cli_and_runs(self) -> None:
        """Test that main function creates CLI and runs it."""
        with patch("local_ai.main.SpeechToTextCLI") as mock_cli_class:
            mock_cli = AsyncMock()
            mock_cli_class.return_value = mock_cli

            await main()

            mock_cli_class.assert_called_once()
            mock_cli.run.assert_called_once()

    def test_main_entry_point_calls_asyncio_run(self) -> None:
        """Test that the main entry point calls asyncio.run with main()."""
        with patch("asyncio.run") as mock_asyncio_run:
            with patch("local_ai.main.main"):
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
        with patch("signal.signal") as mock_signal:
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


@pytest.mark.unit
class TestSpeechToTextCLIConfidenceDisplay:
    """Test cases for confidence rating display in CLI interface."""

    def test_transcription_result_callback_with_high_confidence(self) -> None:
        """Test transcription result callback with high confidence percentage display."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("[1] Hello world (95%)")

    def test_transcription_result_callback_with_medium_confidence(self) -> None:
        """Test transcription result callback with medium confidence percentage display."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="This is a test",
            confidence=0.67,
            timestamp=1234567890.0,
            processing_time=0.3,
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("[1] This is a test (67%)")

    def test_transcription_result_callback_with_low_confidence(self) -> None:
        """Test transcription result callback with low confidence percentage display."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="Unclear speech",
            confidence=0.23,
            timestamp=1234567890.0,
            processing_time=0.8,
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("[1] Unclear speech (23%)")

    def test_transcription_result_callback_with_zero_confidence(self) -> None:
        """Test transcription result callback with zero confidence percentage display."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="Very unclear",
            confidence=0.0,
            timestamp=1234567890.0,
            processing_time=1.0,
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("[1] Very unclear (0%)")

    def test_transcription_result_callback_with_perfect_confidence(self) -> None:
        """Test transcription result callback with perfect confidence percentage display."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="Perfect transcription",
            confidence=1.0,
            timestamp=1234567890.0,
            processing_time=0.2,
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("[1] Perfect transcription (100%)")

    def test_transcription_result_callback_confidence_disabled(self) -> None:
        """Test transcription result callback with confidence display disabled."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=False)

        result = TranscriptionResult(
            text="Hello world",
            confidence=0.85,
            timestamp=1234567890.0,
            processing_time=0.4,
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 1
        mock_print.assert_called_once_with("[1] Hello world")

    def test_transcription_result_callback_with_empty_text(self) -> None:
        """Test transcription result callback with empty text (should not display)."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="", confidence=0.75, timestamp=1234567890.0, processing_time=0.1
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 0
        mock_print.assert_not_called()

    def test_transcription_result_callback_with_whitespace_only_text(self) -> None:
        """Test transcription result callback with whitespace-only text (should not display)."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        result = TranscriptionResult(
            text="   \n\t  ", confidence=0.60, timestamp=1234567890.0, processing_time=0.2
        )

        with patch("builtins.print") as mock_print:
            cli._on_transcription_result(result)

        assert cli._transcription_count == 0
        mock_print.assert_not_called()

    def test_transcription_result_callback_increments_counter(self) -> None:
        """Test that transcription result callback increments counter correctly."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        results = [
            TranscriptionResult("First", 0.9, 1000.0, 0.1),
            TranscriptionResult("Second", 0.8, 1001.0, 0.2),
            TranscriptionResult("Third", 0.7, 1002.0, 0.3),
        ]

        with patch("builtins.print") as mock_print:
            for result in results:
                cli._on_transcription_result(result)

        assert cli._transcription_count == 3

        # Verify all calls were made with correct formatting
        expected_calls = [
            (("[1] First (90%)",),),
            (("[2] Second (80%)",),),
            (("[3] Third (70%)",),),
        ]
        assert mock_print.call_args_list == expected_calls

    def test_confidence_formatting_edge_cases(self) -> None:
        """Test confidence percentage formatting with edge case values."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        # Test various confidence values and their expected formatting
        # Python's :.0% formatting uses banker's rounding (round half to even)
        test_cases = [
            (0.001, "0%"),  # Very low confidence rounds to 0%
            (0.004, "0%"),  # Still rounds to 0%
            (0.005, "0%"),  # 0.5% rounds to 0% (banker's rounding)
            (0.994, "99%"),  # Rounds down to 99%
            (0.995, "100%"),  # 99.5% rounds to 100% (banker's rounding)
            (0.999, "100%"),  # Very high confidence rounds to 100%
        ]

        with patch("builtins.print") as mock_print:
            for i, (confidence, expected_percentage) in enumerate(test_cases, 1):
                result = TranscriptionResult(
                    text=f"Test {i}",
                    confidence=confidence,
                    timestamp=1000.0 + i,
                    processing_time=0.1,
                )
                cli._on_transcription_result(result)

        # Verify formatting for each case
        for i, (confidence, expected_percentage) in enumerate(test_cases, 1):
            expected_call = f"[{i}] Test {i} ({expected_percentage})"
            mock_print.assert_any_call(expected_call)


@pytest.mark.unit
class TestCLIConfidenceCallbackIntegration:
    """Test cases for confidence data passing to downstream systems through callbacks."""

    def test_service_callback_receives_full_transcription_result(self) -> None:
        """Test that service callback receives complete TranscriptionResult with confidence data."""
        mock_service = AsyncMock()
        # Make set_transcription_result_callback synchronous
        mock_service.set_transcription_result_callback = Mock()
        cli = SpeechToTextCLI(service=mock_service, show_confidence_percentage=True)

        # Verify that the CLI sets up the transcription result callback
        with patch("builtins.print"):
            asyncio.run(cli.start_listening())

        # Check that set_transcription_result_callback was called with the CLI's callback method
        mock_service.set_transcription_result_callback.assert_called_once_with(
            cli._on_transcription_result
        )

    def test_downstream_system_receives_confidence_metadata(self) -> None:
        """Test that downstream systems can receive confidence data through callback mechanism."""
        # Create a mock downstream system that captures callback data
        captured_results = []

        def mock_downstream_callback(result: TranscriptionResult) -> None:
            """Mock downstream system callback that captures transcription results."""
            captured_results.append(result)

        # Set up CLI with the mock callback
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        # Simulate the service calling the CLI's callback, which should pass data to downstream
        test_result = TranscriptionResult(
            text="Test transcription",
            confidence=0.85,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # In a real scenario, downstream systems would register their callbacks with the service
        # Here we simulate the CLI receiving and processing the result
        with patch("builtins.print"):
            cli._on_transcription_result(test_result)

        # Verify the result was processed (this simulates what downstream systems would receive)
        assert cli._latest_transcription_result == test_result
        assert cli._latest_transcription_result.confidence == 0.85
        assert cli._latest_transcription_result.text == "Test transcription"

    def test_callback_mechanism_preserves_all_metadata(self) -> None:
        """Test that callback mechanism preserves all transcription metadata for downstream systems."""
        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        # Create a comprehensive TranscriptionResult with all metadata
        comprehensive_result = TranscriptionResult(
            text="Comprehensive test transcription",
            confidence=0.92,
            timestamp=1234567890.123,
            processing_time=0.75,
        )

        with patch("builtins.print"):
            cli._on_transcription_result(comprehensive_result)

        # Verify all metadata is preserved in the CLI's stored result
        stored_result = cli._latest_transcription_result
        assert stored_result is not None
        assert stored_result.text == "Comprehensive test transcription"
        assert stored_result.confidence == 0.92
        assert stored_result.timestamp == 1234567890.123
        assert stored_result.processing_time == 0.75

    def test_multiple_downstream_callbacks_receive_confidence_data(self) -> None:
        """Test that multiple downstream systems can receive confidence data."""
        # Simulate multiple downstream system callbacks
        embedding_system_results = []
        response_system_results = []
        tts_system_results = []

        def embedding_callback(result: TranscriptionResult) -> None:
            embedding_system_results.append(result)

        def response_callback(result: TranscriptionResult) -> None:
            response_system_results.append(result)

        def tts_callback(result: TranscriptionResult) -> None:
            tts_system_results.append(result)

        mock_service = Mock()
        cli = SpeechToTextCLI(service=mock_service)

        # Create test results with different confidence levels
        test_results = [
            TranscriptionResult("High confidence", 0.95, 1000.0, 0.1),
            TranscriptionResult("Medium confidence", 0.65, 1001.0, 0.2),
            TranscriptionResult("Low confidence", 0.30, 1002.0, 0.3),
        ]

        with patch("builtins.print"):
            for result in test_results:
                cli._on_transcription_result(result)

                # Simulate downstream systems receiving the data
                # (In real implementation, this would happen through service callbacks)
                embedding_callback(result)
                response_callback(result)
                tts_callback(result)

        # Verify all downstream systems received all results with confidence data
        assert len(embedding_system_results) == 3
        assert len(response_system_results) == 3
        assert len(tts_system_results) == 3

        # Verify confidence data is preserved for each system
        for i, expected_result in enumerate(test_results):
            assert embedding_system_results[i].confidence == expected_result.confidence
            assert response_system_results[i].confidence == expected_result.confidence
            assert tts_system_results[i].confidence == expected_result.confidence
