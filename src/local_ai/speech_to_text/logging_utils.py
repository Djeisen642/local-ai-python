"""Logging utilities with custom trace level."""

import logging
from typing import Any

# Define custom TRACE level (lower than DEBUG)
TRACE_LEVEL = 5


def add_trace_level() -> None:
    """Add a custom TRACE logging level."""
    # Add the trace level to the logging module
    logging.addLevelName(TRACE_LEVEL, "TRACE")

    # Add trace method to Logger class
    def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a message with severity 'TRACE'."""
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, message, args, **kwargs)

    # Monkey patch the Logger class
    logging.Logger.trace = trace


def get_logger(name: str) -> logging.Logger:
    """Get a logger with trace support."""
    # Ensure trace level is added
    if not hasattr(logging.Logger, "trace"):
        add_trace_level()

    return logging.getLogger(name)
