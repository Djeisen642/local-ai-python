"""Whisper transcription functionality."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import (
    CONFIDENCE_LOGPROB_MAX,
    CONFIDENCE_LOGPROB_MIN,
    DEFAULT_SAMPLE_RATE,
    MAX_AUDIO_FILE_SIZE,
    SILENCE_DURATION,
)
from .logging_utils import get_logger

# Import faster_whisper at module level for proper mocking in tests
try:
    import faster_whisper  # type: ignore[import-untyped]
except ImportError:
    faster_whisper = None

from .cache_utils import get_whisper_cache_dir
from .models import TranscriptionResult

if TYPE_CHECKING:
    from .audio_debugger import AudioDebugger

logger = get_logger(__name__)


class WhisperTranscriber:
    """Uses faster-whisper library for local speech-to-text conversion."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        audio_debugger: AudioDebugger | None = None,
    ) -> None:
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Size of Whisper model to use
            device: Device to use for inference ("cpu" or "cuda")
            compute_type: Compute type for inference ("int8", "float16", etc.)
            audio_debugger: Optional AudioDebugger instance for saving audio files
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None
        self._model_loaded = False
        self._audio_debugger = audio_debugger

    def _load_model(self) -> bool:
        """
        Load the Whisper model with automatic device selection.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded and self._model is not None:
            return True

        try:
            if faster_whisper is None:
                raise ImportError("faster-whisper library not available")

            WhisperModel = faster_whisper.WhisperModel

            # Use configured device and compute type
            device = self.device
            compute_type = self.compute_type

            logger.debug(
                f"Loading Whisper '{self.model_size}' on {device} ({compute_type})"
            )

            # Use unified cache directory for Whisper models
            whisper_cache_dir = get_whisper_cache_dir()

            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(whisper_cache_dir),
            )

            self._model_loaded = True
            logger.debug(f"Successfully loaded Whisper model '{self.model_size}'")
            return True

        except ImportError as e:
            logger.error(f"faster-whisper library not available: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"Whisper model files not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def _convert_audio_format(
        self,
        audio_data: bytes,
        target_sample_rate: int | None = None,
        source_sample_rate: int | None = None,
    ) -> bytes:
        """
        Convert audio data to format compatible with Whisper.

        Args:
            audio_data: Raw audio data
            target_sample_rate: Target sample rate for conversion
            source_sample_rate: Source sample rate of the audio data

        Returns:
            Converted audio data in WAV format
        """
        if target_sample_rate is None:
            target_sample_rate = DEFAULT_SAMPLE_RATE

        if source_sample_rate is None:
            source_sample_rate = DEFAULT_SAMPLE_RATE

        if not audio_data or not isinstance(audio_data, bytes):
            return b""

        # If already WAV format, return as-is (Whisper handles conversion)
        if audio_data.startswith(b"RIFF") and b"WAVE" in audio_data:
            return audio_data

        # For raw audio data, create a basic WAV file
        # Assume 16-bit mono audio at source sample rate
        return self._create_wav_data(
            audio_data, source_sample_rate, 1, target_sample_rate
        )

    def _create_wav_data(
        self,
        audio_samples: bytes,
        source_sample_rate: int,
        channels: int,
        target_sample_rate: int,
    ) -> bytes:
        """
        Create WAV file data from raw audio samples.

        Args:
            audio_samples: Raw audio sample data
            source_sample_rate: Original sample rate
            channels: Number of audio channels
            target_sample_rate: Target sample rate

        Returns:
            WAV file data as bytes
        """
        try:
            # Import config for debug logging flag
            from .config import AUDIO_DEBUG_LOG_SAMPLE_RATES

            # Convert to numpy array for processing
            if channels == 1:
                # Mono audio - 16-bit signed integers
                samples = np.frombuffer(audio_samples, dtype=np.int16)
            else:
                # Multi-channel audio - convert to mono by averaging channels
                samples = np.frombuffer(audio_samples, dtype=np.int16)
                samples = samples.reshape(-1, channels)
                samples = np.mean(samples, axis=1).astype(np.int16)

            # Resample if needed (skip when source == target)
            if source_sample_rate != target_sample_rate:
                if AUDIO_DEBUG_LOG_SAMPLE_RATES:
                    logger.debug(
                        f"Resampling audio: {source_sample_rate}Hz -> {target_sample_rate}Hz"
                    )
                samples = self._resample_audio(
                    samples, source_sample_rate, target_sample_rate
                )
            elif AUDIO_DEBUG_LOG_SAMPLE_RATES:
                logger.debug(
                    f"Skipping resampling: source and target both {source_sample_rate}Hz"
                )

            # Create WAV file in memory
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(target_sample_rate)
                wav_file.writeframes(samples.tobytes())

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to create WAV data: {e}")
            # Return a minimal valid WAV file if conversion fails
            return self._create_minimal_wav()

    def _resample_audio(
        self, samples: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """
        Simple audio resampling using linear interpolation.

        Args:
            samples: Audio samples as numpy array
            source_rate: Original sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio samples
        """
        if source_rate == target_rate:
            return samples

        # Calculate resampling ratio
        ratio = target_rate / source_rate
        new_length = int(len(samples) * ratio)

        # Simple linear interpolation resampling
        old_indices = np.linspace(0, len(samples) - 1, new_length)
        new_samples = np.interp(old_indices, np.arange(len(samples)), samples)

        return new_samples.astype(np.int16)

    def _create_minimal_wav(self) -> bytes:
        """
        Create a minimal valid WAV file with silence.

        Returns:
            Minimal WAV file data
        """
        # Create silence at default sample rate
        sample_rate = DEFAULT_SAMPLE_RATE
        duration = SILENCE_DURATION
        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        return buffer.getvalue()

    def _calculate_confidence(self, segments: list) -> float:
        """
        Convert faster-whisper avg_logprob to normalized confidence score (0.0-1.0).

        avg_logprob typically ranges from -2.0 (low confidence) to -0.1 (high confidence)
        We normalize this to a 0.0-1.0 scale for user-friendly display.

        Args:
            segments: List of transcription segments from faster-whisper

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not segments:
            return 0.0

        # Calculate weighted average of segment confidences
        total_duration = 0.0
        weighted_logprob = 0.0

        for segment in segments:
            if (
                hasattr(segment, "avg_logprob")
                and hasattr(segment, "start")
                and hasattr(segment, "end")
            ):
                duration = segment.end - segment.start
                total_duration += duration
                weighted_logprob += segment.avg_logprob * duration

        if total_duration == 0:
            return 0.0

        avg_logprob = weighted_logprob / total_duration

        # Convert log probability to confidence (0.0-1.0)
        # Typical range: avg_logprob from -2.0 to -0.1
        # Normalize to 0.0-1.0 where higher logprob = higher confidence
        return max(
            0.0,
            min(
                1.0,
                (avg_logprob - CONFIDENCE_LOGPROB_MIN)
                / (CONFIDENCE_LOGPROB_MAX - CONFIDENCE_LOGPROB_MIN),
            ),
        )

    def _post_process_text(self, text: str) -> str:
        """
        Post-process transcribed text for better formatting.

        Args:
            text: Raw transcribed text

        Returns:
            Cleaned and formatted text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove leading/trailing whitespace
        processed = text.strip()

        # Handle empty result
        if not processed:
            return ""

        # Basic text normalization
        # Remove excessive whitespace while preserving single spaces
        import re

        processed = re.sub(r"\s+", " ", processed)

        # Ensure proper sentence capitalization
        if processed and processed[0].islower():
            processed = processed[0].upper() + processed[1:]

        return processed

    def create_transcription_result(
        self, text: str, processing_start_time: float, confidence: float = 0.0
    ) -> TranscriptionResult:
        """
        Create a TranscriptionResult object from transcription data.

        Args:
            text: Transcribed text
            processing_start_time: When processing started (for timing calculation)
            confidence: Confidence score (if available)

        Returns:
            TranscriptionResult object
        """
        current_time = time.time()
        processing_time = current_time - processing_start_time

        return TranscriptionResult(
            text=self._post_process_text(text),
            confidence=confidence,
            timestamp=current_time,
            processing_time=processing_time,
        )

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data to transcribe

        Returns:
            Transcribed text
        """
        # Handle invalid input
        if not audio_data or not isinstance(audio_data, bytes):
            logger.debug("Transcriber received invalid audio data")
            return ""

        # Handle empty audio
        if len(audio_data) == 0:
            logger.debug("Transcriber received empty audio data")
            return ""

        # Handle excessively large audio data
        if len(audio_data) > MAX_AUDIO_FILE_SIZE:
            logger.warning(
                f"Audio data too large ({len(audio_data)} bytes), skipping transcription"
            )
            return ""

        logger.debug(f"🎤 Transcriber processing {len(audio_data)} bytes of audio data")
        processing_start_time = time.time()

        try:
            # Load model if not already loaded
            if not self._load_model() or self._model is None:
                logger.warning("Whisper model not available for transcription")
                return ""

            # Convert audio to Whisper-compatible format
            logger.debug(
                f"🔄 Converting audio for Whisper (target: {DEFAULT_SAMPLE_RATE}Hz)"
            )
            converted_audio = self._convert_audio_format(
                audio_data, target_sample_rate=DEFAULT_SAMPLE_RATE
            )

            if not converted_audio:
                logger.warning("❌ Audio format conversion failed")
                return ""

            logger.debug(f"✅ Audio converted successfully: {len(converted_audio)} bytes")

            # Create a temporary file for faster-whisper to process
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(converted_audio)
                temp_file.flush()

                try:
                    logger.debug(f"🔄 Starting Whisper transcription: {temp_file.name}")

                    # Run transcription in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    segments, _ = await loop.run_in_executor(
                        None, self._model.transcribe, temp_file.name
                    )

                    # Collect all text segments
                    transcribed_text = ""
                    segment_count = 0
                    for segment in segments:
                        transcribed_text += segment.text
                        segment_count += 1

                    logger.debug(
                        f"📝 Whisper returned {segment_count} segments, "
                        f"raw text length: {len(transcribed_text)}"
                    )

                    # Post-process the transcribed text
                    processed_text = self._post_process_text(transcribed_text)

                    processing_time = time.time() - processing_start_time

                    if processed_text.strip():
                        logger.trace(
                            f"✅ Transcription successful: '{processed_text}' "
                            f"({len(audio_data)} bytes -> {len(processed_text)} chars, "
                            f"{processing_time:.2f}s)"
                        )
                    else:
                        logger.debug(
                            f"🔇 Transcription returned empty result after processing "
                            f"({processing_time:.2f}s)"
                        )

                    return processed_text

                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass  # File might already be deleted

        except TimeoutError as e:
            logger.error(f"Transcription timeout: {e}")
            return ""
        except MemoryError as e:
            logger.error(f"Transcription memory error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    async def transcribe_audio_with_result(
        self, audio_data: bytes, source_sample_rate: int | None = None
    ) -> TranscriptionResult:
        """
        Transcribe audio data and return detailed result information.

        Args:
            audio_data: Audio data to transcribe
            source_sample_rate: Source sample rate of the audio data

        Returns:
            TranscriptionResult with text, confidence, and timing information
        """
        processing_start_time = time.time()

        # Handle invalid input
        if not audio_data or not isinstance(audio_data, bytes) or len(audio_data) == 0:
            return self.create_transcription_result("", processing_start_time, 0.0)

        # Handle excessively large audio data
        if len(audio_data) > MAX_AUDIO_FILE_SIZE:
            logger.warning(
                f"Audio data too large ({len(audio_data)} bytes), skipping transcription"
            )
            return self.create_transcription_result("", processing_start_time, 0.0)

        try:
            # Load model if not already loaded
            if not self._load_model() or self._model is None:
                logger.warning("Whisper model not available for transcription")
                return self.create_transcription_result("", processing_start_time, 0.0)

            # Convert audio to Whisper-compatible format
            converted_audio = self._convert_audio_format(
                audio_data,
                target_sample_rate=DEFAULT_SAMPLE_RATE,
                source_sample_rate=source_sample_rate,
            )

            if not converted_audio:
                logger.warning("Audio format conversion failed")
                return self.create_transcription_result("", processing_start_time, 0.0)

            # Debug: Save audio if debugging is enabled
            if self._audio_debugger is not None and self._audio_debugger.is_enabled():
                try:
                    # Extract raw audio data from WAV format for saving
                    # The converted_audio is in WAV format, extract the raw PCM data
                    import wave as wave_module

                    with io.BytesIO(converted_audio) as wav_buffer:
                        with wave_module.open(wav_buffer, "rb") as wav_file:
                            raw_audio_data = wav_file.readframes(wav_file.getnframes())

                    saved_path = self._audio_debugger.save_audio_sync(
                        raw_audio_data, sample_rate=DEFAULT_SAMPLE_RATE
                    )
                    if saved_path:
                        logger.debug(f"Audio debug file saved: {saved_path}")
                except Exception as e:
                    logger.debug(f"Audio debug save failed: {e}")

            # Create a temporary file for faster-whisper to process
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(converted_audio)
                temp_file.flush()

                try:
                    # Run transcription in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    segments, info = await loop.run_in_executor(
                        None, self._model.transcribe, temp_file.name
                    )

                    # Collect all text segments and calculate confidence
                    transcribed_text = ""
                    segments_list = list(segments)

                    for segment in segments_list:
                        transcribed_text += segment.text

                    # Calculate confidence using proper method
                    confidence = self._calculate_confidence(segments_list)

                    return self.create_transcription_result(
                        transcribed_text, processing_start_time, confidence
                    )

                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass  # File might already be deleted

        except TimeoutError as e:
            logger.error(f"Transcription timeout: {e}")
            return self.create_transcription_result("", processing_start_time, 0.0)
        except MemoryError as e:
            logger.error(f"Transcription memory error: {e}")
            return self.create_transcription_result("", processing_start_time, 0.0)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return self.create_transcription_result("", processing_start_time, 0.0)

    def is_model_available(self) -> bool:
        """
        Check if the Whisper model is available.

        Returns:
            True if model is available, False otherwise
        """
        try:
            return self._load_model()
        except Exception as e:
            logger.debug(f"Model availability check failed: {e}")
            return False

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        if not self.is_model_available() or self._model is None:
            return {}

        try:
            info = {
                "model_size": self.model_size,
                "model_loaded": self._model_loaded,
            }

            # Try to get additional model information if available
            if hasattr(self._model, "model_size"):
                info["actual_model_size"] = self._model.model_size
            if hasattr(self._model, "device"):
                info["device"] = str(self._model.device)
            if hasattr(self._model, "compute_type"):
                info["compute_type"] = str(self._model.compute_type)

            return info

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    def clear_model_cache(self) -> bool:
        """
        Clear the downloaded model cache and reset the transcriber.

        This removes cached model files from the HuggingFace cache directory
        and resets the transcriber state so models will be re-downloaded on next use.

        Returns:
            True if cache was cleared successfully, False otherwise
        """
        try:
            import shutil
            from pathlib import Path

            # Reset internal state first
            self._model = None
            self._model_loaded = False

            models_cleared = 0

            # Clear unified cache directory
            whisper_cache_dir = get_whisper_cache_dir()
            if whisper_cache_dir.exists():
                try:
                    logger.debug(
                        f"Removing unified Whisper cache directory: {whisper_cache_dir}"
                    )
                    shutil.rmtree(whisper_cache_dir)
                    models_cleared += 1
                    # Recreate the directory for future use
                    whisper_cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove cache directory {whisper_cache_dir}: {e}"
                    )

            # Also clear legacy HuggingFace cache directory for backward compatibility
            hf_cache_dir = Path.home() / ".cache" / "huggingface"
            if hf_cache_dir.exists():
                # Check for faster-whisper cache directories
                for item in hf_cache_dir.iterdir():
                    if item.is_dir() and (
                        "whisper" in item.name.lower() or "openai" in item.name.lower()
                    ):
                        try:
                            logger.debug(
                                f"Removing legacy cached model directory: {item}"
                            )
                            shutil.rmtree(item)
                            models_cleared += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove legacy cache directory {item}: {e}"
                            )

                # Also check hub cache for transformers-style models
                hub_cache = hf_cache_dir / "hub"
                if hub_cache.exists():
                    for item in hub_cache.iterdir():
                        if item.is_dir() and "whisper" in item.name.lower():
                            try:
                                logger.debug(f"Removing legacy cached hub model: {item}")
                                shutil.rmtree(item)
                                models_cleared += 1
                            except Exception as e:
                                logger.warning(
                                    f"Failed to remove legacy hub cache {item}: {e}"
                                )

            logger.info(f"Cleared {models_cleared} cached model directories")
            return True

        except Exception as e:
            logger.error(f"Failed to clear model cache: {e}")
            return False
