"""Custom exceptions for speech-to-text functionality."""


class SpeechToTextError(Exception):
    """Base exception for speech-to-text errors."""

    pass


class AudioCaptureError(SpeechToTextError):
    """Exception raised for audio capture related errors."""

    pass


class MicrophoneNotFoundError(AudioCaptureError):
    """Exception raised when no microphone is found."""

    pass


class TranscriptionError(SpeechToTextError):
    """Exception raised for transcription related errors."""

    pass
