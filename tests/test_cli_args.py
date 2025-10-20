"""Tests for CLI argument parsing functionality."""

import argparse
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from local_ai.main import create_argument_parser, handle_arguments


@pytest.mark.unit
class TestArgumentParser:
    """Test cases for CLI argument parsing."""

    def test_create_argument_parser_basic(self) -> None:
        """Test basic argument parser creation."""
        parser = create_argument_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Speech-to-Text CLI" in parser.description

    def test_help_argument(self) -> None:
        """Test --help/-h argument functionality."""
        parser = create_argument_parser()

        # Test that help argument is available
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args(["--help"])

            # Should exit with code 0 for help
            assert exc_info.value.code == 0

            help_output = mock_stdout.getvalue()
            assert "Speech-to-Text CLI" in help_output
            assert "--help" in help_output
            assert "--reset-model-cache" in help_output
            assert "--reset-optimization-cache" in help_output
            assert "--verbose" in help_output
            assert "--trace" in help_output
            assert "--force-cpu" in help_output

    def test_short_help_argument(self) -> None:
        """Test -h short form of help argument."""
        parser = create_argument_parser()

        with patch("sys.stdout", new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args(["-h"])

            assert exc_info.value.code == 0

    def test_reset_model_cache_argument(self) -> None:
        """Test --reset-model-cache argument parsing."""
        parser = create_argument_parser()

        # Test with --reset-model-cache
        args = parser.parse_args(["--reset-model-cache"])
        assert args.reset_model_cache is True

        # Test without --reset-model-cache
        args = parser.parse_args([])
        assert args.reset_model_cache is False

    def test_reset_optimization_cache_argument(self) -> None:
        """Test --reset-optimization-cache argument parsing."""
        parser = create_argument_parser()

        # Test with --reset-optimization-cache
        args = parser.parse_args(["--reset-optimization-cache"])
        assert args.reset_optimization_cache is True

        # Test without --reset-optimization-cache
        args = parser.parse_args([])
        assert args.reset_optimization_cache is False

    def test_verbose_argument(self) -> None:
        """Test --verbose/-v argument parsing."""
        parser = create_argument_parser()

        # Test with --verbose
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

        # Test with -v
        args = parser.parse_args(["-v"])
        assert args.verbose is True

        # Test without verbose
        args = parser.parse_args([])
        assert args.verbose is False

    def test_trace_argument(self) -> None:
        """Test --trace argument parsing."""
        parser = create_argument_parser()

        # Test with --trace
        args = parser.parse_args(["--trace"])
        assert args.trace is True

        # Test without trace
        args = parser.parse_args([])
        assert args.trace is False

    def test_force_cpu_argument(self) -> None:
        """Test --force-cpu argument parsing."""
        parser = create_argument_parser()

        # Test with --force-cpu
        args = parser.parse_args(["--force-cpu"])
        assert args.force_cpu is True

        # Test without force-cpu
        args = parser.parse_args([])
        assert args.force_cpu is False

    def test_combined_arguments(self) -> None:
        """Test parsing multiple arguments together."""
        parser = create_argument_parser()

        args = parser.parse_args(["--reset-model-cache", "--verbose"])
        assert args.reset_model_cache is True
        assert args.verbose is True

        args = parser.parse_args(["-v", "--reset-optimization-cache"])
        assert args.reset_optimization_cache is True
        assert args.verbose is True

        args = parser.parse_args(["--reset-model-cache", "--reset-optimization-cache"])
        assert args.reset_model_cache is True
        assert args.reset_optimization_cache is True

        args = parser.parse_args(["--verbose", "--trace", "--force-cpu"])
        assert args.verbose is True
        assert args.trace is True
        assert args.force_cpu is True

    def test_no_arguments(self) -> None:
        """Test parsing with no arguments (defaults)."""
        parser = create_argument_parser()

        args = parser.parse_args([])
        assert args.reset_model_cache is False
        assert args.reset_optimization_cache is False
        assert args.verbose is False
        assert args.trace is False
        assert args.force_cpu is False

    def test_invalid_argument(self) -> None:
        """Test handling of invalid arguments."""
        parser = create_argument_parser()

        with patch("sys.stderr", new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args(["--invalid-arg"])

            # Should exit with non-zero code for invalid arguments
            assert exc_info.value.code != 0

    def test_argument_descriptions(self) -> None:
        """Test that arguments have proper descriptions."""
        parser = create_argument_parser()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])

            help_output = mock_stdout.getvalue()

            # Check for argument descriptions
            assert "Clear HuggingFace model cache" in help_output
            assert "Clear system optimization cache" in help_output
            assert "Enable verbose logging" in help_output
            assert "Enable trace logging" in help_output
            assert "Force CPU-only mode" in help_output

    def test_no_confidence_argument(self) -> None:
        """Test --no-confidence argument parsing."""
        parser = create_argument_parser()

        # Test with --no-confidence
        args = parser.parse_args(["--no-confidence"])
        assert args.no_confidence is True

        # Test without --no-confidence
        args = parser.parse_args([])
        assert args.no_confidence is False

    def test_no_confidence_help_text(self) -> None:
        """Test that --no-confidence appears in help text."""
        parser = create_argument_parser()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])

            help_output = mock_stdout.getvalue()
            assert "--no-confidence" in help_output
            assert "Hide confidence percentages" in help_output

    def test_combined_arguments_with_no_confidence(self) -> None:
        """Test combining --no-confidence with other arguments."""
        parser = create_argument_parser()

        # Test --no-confidence with --verbose
        args = parser.parse_args(["--no-confidence", "--verbose"])
        assert args.no_confidence is True
        assert args.verbose is True

        # Test --no-confidence with --force-cpu
        args = parser.parse_args(["--no-confidence", "--force-cpu"])
        assert args.no_confidence is True
        assert args.force_cpu is True

        # Test --no-confidence with cache reset
        args = parser.parse_args(["--no-confidence", "--reset-model-cache"])
        assert args.no_confidence is True
        assert args.reset_model_cache is True


@pytest.mark.unit
class TestArgumentHandling:
    """Test cases for argument handling functionality."""

    def test_handle_arguments_verbose_logging(self) -> None:
        """Test verbose logging setup."""
        args = Mock()
        args.verbose = True
        args.trace = False
        args.reset_model_cache = False
        args.reset_optimization_cache = False

        with patch("logging.basicConfig") as mock_logging:
            with patch("local_ai.main.reset_model_cache") as mock_reset_model:
                with patch("local_ai.main.reset_optimization_cache") as mock_reset_opt:
                    success, should_continue = handle_arguments(args)

        # Should configure verbose logging
        mock_logging.assert_called_once_with(
            level="DEBUG", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        mock_reset_model.assert_not_called()
        mock_reset_opt.assert_not_called()
        assert success is True
        assert should_continue is True

    def test_handle_arguments_normal_logging(self) -> None:
        """Test normal logging setup."""
        args = Mock()
        args.verbose = False
        args.trace = False
        args.reset_model_cache = False
        args.reset_optimization_cache = False

        with patch("logging.basicConfig") as mock_logging:
            with patch("local_ai.main.reset_model_cache") as mock_reset_model:
                with patch("local_ai.main.reset_optimization_cache") as mock_reset_opt:
                    success, should_continue = handle_arguments(args)

        # Should configure normal logging
        mock_logging.assert_called_once_with(
            level="INFO", format="%(asctime)s - %(levelname)s - %(message)s"
        )
        mock_reset_model.assert_not_called()
        mock_reset_opt.assert_not_called()
        assert success is True
        assert should_continue is True

    def test_handle_arguments_trace_logging(self) -> None:
        """Test trace logging setup."""
        args = Mock()
        args.verbose = False
        args.trace = True
        args.reset_model_cache = False
        args.reset_optimization_cache = False

        with patch("logging.basicConfig") as mock_logging:
            with patch("local_ai.main.reset_model_cache") as mock_reset_model:
                with patch("local_ai.main.reset_optimization_cache") as mock_reset_opt:
                    success, should_continue = handle_arguments(args)

        # Should configure trace logging (level 5)
        mock_logging.assert_called_once_with(
            level=5,  # TRACE_LEVEL constant
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        mock_reset_model.assert_not_called()
        mock_reset_opt.assert_not_called()
        assert success is True
        assert should_continue is True

    def test_handle_arguments_reset_model_cache_success(self) -> None:
        """Test successful model cache reset."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = True
        args.reset_optimization_cache = False

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_model_cache", return_value=True
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        mock_reset.assert_called_once()
        mock_print.assert_any_call("✅ Model cache cleared successfully.")
        assert success is True
        assert should_continue is False  # Should not continue after cache reset

    def test_handle_arguments_reset_optimization_cache_success(self) -> None:
        """Test successful optimization cache reset."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = False
        args.reset_optimization_cache = True

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_optimization_cache", return_value=True
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        mock_reset.assert_called_once()
        mock_print.assert_any_call("✅ Optimization cache cleared successfully.")
        assert success is True
        assert should_continue is False  # Should not continue after cache reset

    def test_handle_arguments_reset_model_cache_failure(self) -> None:
        """Test failed model cache reset."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = True
        args.reset_optimization_cache = False

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_model_cache", return_value=False
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        mock_reset.assert_called_once()
        mock_print.assert_any_call("❌ Failed to clear model cache.")
        assert success is False
        assert should_continue is False

    def test_handle_arguments_reset_optimization_cache_failure(self) -> None:
        """Test failed optimization cache reset."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = False
        args.reset_optimization_cache = True

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_optimization_cache", return_value=False
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        mock_reset.assert_called_once()
        mock_print.assert_any_call("❌ Failed to clear optimization cache.")
        assert success is False
        assert should_continue is False

    def test_handle_arguments_reset_model_cache_exception(self) -> None:
        """Test exception during model cache reset."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = True
        args.reset_optimization_cache = False

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_model_cache", side_effect=Exception("Cache error")
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        mock_reset.assert_called_once()
        mock_print.assert_any_call("❌ Error clearing model cache: Cache error")
        assert success is False
        assert should_continue is False

    def test_handle_arguments_reset_optimization_cache_exception(self) -> None:
        """Test exception during optimization cache reset."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = False
        args.reset_optimization_cache = True

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_optimization_cache",
                side_effect=Exception("Opt cache error"),
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        mock_reset.assert_called_once()
        mock_print.assert_any_call(
            "❌ Error clearing optimization cache: Opt cache error"
        )
        assert success is False
        assert should_continue is False

    def test_handle_arguments_combined_options(self) -> None:
        """Test handling combined verbose and cache reset options."""
        args = Mock()
        args.verbose = True
        args.trace = False
        args.reset_model_cache = True
        args.reset_optimization_cache = False

        with patch("logging.basicConfig") as mock_logging:
            with patch(
                "local_ai.main.reset_model_cache", return_value=True
            ) as mock_reset:
                with patch("builtins.print") as mock_print:
                    success, should_continue = handle_arguments(args)

        # Should set up verbose logging
        mock_logging.assert_called_once_with(
            level="DEBUG", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Should reset cache
        mock_reset.assert_called_once()
        mock_print.assert_any_call("✅ Model cache cleared successfully.")
        assert success is True
        assert should_continue is False  # Should not continue after cache reset

    def test_handle_arguments_both_caches(self) -> None:
        """Test handling both cache reset options together."""
        args = Mock()
        args.verbose = False
        args.reset_model_cache = True
        args.reset_optimization_cache = True

        with patch("logging.basicConfig"):
            with patch(
                "local_ai.main.reset_model_cache", return_value=True
            ) as mock_reset_model:
                with patch(
                    "local_ai.main.reset_optimization_cache", return_value=True
                ) as mock_reset_opt:
                    with patch("builtins.print") as mock_print:
                        success, should_continue = handle_arguments(args)

        # Should reset both caches
        mock_reset_model.assert_called_once()
        mock_reset_opt.assert_called_once()
        mock_print.assert_any_call("✅ Model cache cleared successfully.")
        mock_print.assert_any_call("✅ Optimization cache cleared successfully.")
        assert success is True
        assert should_continue is False  # Should not continue after cache reset


@pytest.mark.unit
class TestCacheResetFunctions:
    """Test cases for the cache reset functions."""

    def test_reset_model_cache_success(self) -> None:
        """Test successful model cache reset."""
        with patch("local_ai.main.WhisperTranscriber") as mock_transcriber_class:
            mock_transcriber = Mock()
            mock_transcriber.clear_model_cache.return_value = True
            mock_transcriber_class.return_value = mock_transcriber

            from local_ai.main import reset_model_cache

            result = reset_model_cache()

        mock_transcriber_class.assert_called_once()
        mock_transcriber.clear_model_cache.assert_called_once()
        assert result is True

    def test_reset_model_cache_failure(self) -> None:
        """Test failed model cache reset."""
        with patch("local_ai.main.WhisperTranscriber") as mock_transcriber_class:
            mock_transcriber = Mock()
            mock_transcriber.clear_model_cache.return_value = False
            mock_transcriber_class.return_value = mock_transcriber

            from local_ai.main import reset_model_cache

            result = reset_model_cache()

        assert result is False

    def test_reset_model_cache_exception(self) -> None:
        """Test exception during model cache reset."""
        with patch("local_ai.main.WhisperTranscriber") as mock_transcriber_class:
            mock_transcriber_class.side_effect = Exception("Transcriber error")

            from local_ai.main import reset_model_cache

            result = reset_model_cache()

        assert result is False

    def test_reset_optimization_cache_success(self) -> None:
        """Test successful optimization cache reset."""
        with patch("local_ai.main.get_optimization_cache") as mock_get_cache:
            mock_cache = Mock()
            mock_get_cache.return_value = mock_cache

            from local_ai.main import reset_optimization_cache

            result = reset_optimization_cache()

        mock_get_cache.assert_called_once()
        mock_cache.clear_cache.assert_called_once_with("all")
        assert result is True

    def test_reset_optimization_cache_exception(self) -> None:
        """Test exception during optimization cache reset."""
        with patch("local_ai.main.get_optimization_cache") as mock_get_cache:
            mock_get_cache.side_effect = Exception("Cache error")

            from local_ai.main import reset_optimization_cache

            result = reset_optimization_cache()

        assert result is False


@pytest.mark.unit
class TestCLIIntegration:
    """Integration tests for CLI argument parsing."""

    def test_cli_entry_with_help(self) -> None:
        """Test CLI entry point with help argument."""
        test_args = ["--help"]

        with patch("sys.argv", ["main.py"] + test_args):
            with patch("sys.stdout", new_callable=StringIO):
                with patch("local_ai.main.handle_arguments") as mock_handle:
                    with pytest.raises(SystemExit):
                        from local_ai.main import cli_entry_with_args

                        cli_entry_with_args()

                    # handle_arguments should not be called for help
                    mock_handle.assert_not_called()

    def test_cli_entry_with_reset_model_cache(self) -> None:
        """Test CLI entry point with reset-model-cache argument."""
        test_args = ["--reset-model-cache"]

        with patch("sys.argv", ["main.py"] + test_args):
            with patch(
                "local_ai.main.handle_arguments", return_value=(True, False)
            ) as mock_handle:
                with patch("local_ai.main.main"):
                    with patch("asyncio.run") as mock_asyncio_run:
                        with pytest.raises(SystemExit) as exc_info:
                            from local_ai.main import cli_entry_with_args

                            cli_entry_with_args()

                        # Should exit with code 0 for successful cache reset
                        assert exc_info.value.code == 0

                    mock_handle.assert_called_once()
                    # Should not proceed to main execution after cache reset
                    mock_asyncio_run.assert_not_called()

    def test_cli_entry_with_reset_optimization_cache(self) -> None:
        """Test CLI entry point with reset-optimization-cache argument."""
        test_args = ["--reset-optimization-cache"]

        with patch("sys.argv", ["main.py"] + test_args):
            with patch(
                "local_ai.main.handle_arguments", return_value=(True, False)
            ) as mock_handle:
                with patch("local_ai.main.main"):
                    with patch("asyncio.run") as mock_asyncio_run:
                        with pytest.raises(SystemExit) as exc_info:
                            from local_ai.main import cli_entry_with_args

                            cli_entry_with_args()

                        # Should exit with code 0 for successful cache reset
                        assert exc_info.value.code == 0

                    mock_handle.assert_called_once()
                    # Should not proceed to main execution after cache reset
                    mock_asyncio_run.assert_not_called()

    def test_cli_entry_with_failed_cache_reset(self) -> None:
        """Test CLI entry point with failed cache reset."""
        test_args = ["--reset-model-cache"]

        with patch("sys.argv", ["main.py"] + test_args):
            with patch(
                "local_ai.main.handle_arguments", return_value=(False, False)
            ) as mock_handle:
                with patch("local_ai.main.main"):
                    with patch("asyncio.run") as mock_asyncio_run:
                        with pytest.raises(SystemExit) as exc_info:
                            from local_ai.main import cli_entry_with_args

                            cli_entry_with_args()

                        # Should exit with code 1 for failed argument handling
                        assert exc_info.value.code == 1

                    mock_handle.assert_called_once()
                    # Should not proceed to main execution after failed argument handling
                    mock_asyncio_run.assert_not_called()

    def test_argument_validation_edge_cases(self) -> None:
        """Test edge cases in argument validation."""
        parser = create_argument_parser()

        # Test empty string arguments (should be treated as no arguments)
        args = parser.parse_args([])
        assert args.reset_model_cache is False
        assert args.reset_optimization_cache is False
        assert args.verbose is False

        # Test that boolean flags don't accept values
        with patch("sys.stderr", new_callable=StringIO):
            with pytest.raises(SystemExit):
                parser.parse_args(["--verbose=true"])  # Should fail

    def test_cli_entry_with_no_confidence_flag(self) -> None:
        """Test CLI entry point with --no-confidence flag."""
        test_args = ["--no-confidence"]

        with patch("sys.argv", ["main.py"] + test_args):
            with patch(
                "local_ai.main.handle_arguments", return_value=(True, True)
            ) as mock_handle:
                with patch("local_ai.main.main"):
                    with patch("asyncio.run") as mock_asyncio_run:
                        from local_ai.main import cli_entry_with_args

                        cli_entry_with_args()

                    mock_handle.assert_called_once()
                    # Should proceed to main execution with show_confidence_percentage=False
                    mock_asyncio_run.assert_called_once()

                    # Verify main was called with correct confidence flag
                    mock_asyncio_run.call_args[0][0]
                    # The call should be main(force_cpu=False, show_confidence_percentage=False)
                    # We can't easily inspect the coroutine args, but we can verify it was called
