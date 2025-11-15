"""Main speech-to-text service orchestrator."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from . import config
from .audio_capture import AudioCapture, AudioCaptureError
from .config import DEFAULT_SAMPLE_RATE, ERROR_RECOVERY_SLEEP
from .interfaces import ProcessingHandler, ProcessingResult
from .logging_utils import get_logger
from .models import TranscriptionResult
from .optimization import get_optimizer
from .performance_monitor import PerformanceContext, get_performance_monitor
from .pipeline import PluginProcessingPipeline, create_processing_context
from .transcriber import WhisperTranscriber
from .vad import VoiceActivityDetector

logger = get_logger(__name__)


class SpeechToTextService:
    """Main orchestrator that coordinates all speech-to-text components."""

    def __init__(
        self,
        optimization_target: str = "balanced",
        enable_monitoring: bool = False,
        use_cache: bool = True,
        force_cpu: bool = False,
        enable_filtering: bool = False,
        enable_task_detection: bool = False,
        task_detection_service: Any | None = None,
        enable_audio_debugging: bool = False,
        audio_debug_dir: Any | None = None,
    ) -> None:
        """
        Initialize the speech-to-text service.

        Args:
            optimization_target: "latency", "accuracy", "resource", or "balanced"
            enable_monitoring: Whether to enable performance monitoring (adds overhead)
            use_cache: Whether to use cached optimization data
            force_cpu: Whether to force CPU-only mode (disable GPU/CUDA)
            enable_filtering: Whether to enable audio filtering pipeline (disabled by
                default due to performance overhead with minimal accuracy improvement
                on clean audio - see docs/audio-filtering-evaluation.md)
            enable_task_detection: Whether to enable task detection from transcriptions
            task_detection_service: Optional TaskDetectionService instance for task
                detection (if None and enable_task_detection is True, will be created)
            enable_audio_debugging: Whether to enable audio debugging (saves audio to WAV)
            audio_debug_dir: Optional directory for audio debug files (None = default)
        """
        self._listening = False
        self._transcription_result_callback: (
            Callable[[TranscriptionResult], None] | None
        ) = None
        self._latest_transcription_result: TranscriptionResult | None = None

        # Initialize components
        self._audio_capture: AudioCapture | None = None
        self._vad: VoiceActivityDetector | None = None
        self._transcriber: WhisperTranscriber | None = None

        # Audio filtering
        self._enable_filtering = enable_filtering
        self._audio_filter_pipeline: Any | None = None

        # Task detection integration
        self._enable_task_detection = enable_task_detection
        self._task_detection_service = task_detection_service

        # Audio debugging
        self._enable_audio_debugging = enable_audio_debugging
        self._audio_debug_dir = audio_debug_dir
        self._audio_debugger: Any | None = None

        # Processing state
        self._processing_task: asyncio.Task | None = None

        # Performance optimization
        self._use_cache = use_cache
        self._force_cpu = force_cpu
        self._optimizer = get_optimizer(use_cache=use_cache, force_cpu=force_cpu)
        self._optimization_target = optimization_target
        self._optimized_config = self._get_optimized_config()

        # Optional performance monitoring
        self._monitoring_enabled = enable_monitoring
        self._performance_monitor = (
            get_performance_monitor() if enable_monitoring else None
        )

        # Plugin processing pipeline for future system integration
        self._processing_pipeline = PluginProcessingPipeline()
        self._pipeline_callback: Callable[[list[ProcessingResult]], None] | None = None

    def _get_optimized_config(self) -> dict:
        """Get optimized configuration based on target."""
        from .optimization import get_optimized_config

        return get_optimized_config(
            self._optimization_target,
            use_cache=self._use_cache,
            force_cpu=self._force_cpu,
        )

    def _initialize_components(self) -> bool:
        """
        Initialize all speech-to-text components with optimized configuration.

        Returns:
            True if all components initialized successfully, False otherwise
        """
        try:
            # Get optimized configurations
            audio_config = self._optimizer.get_optimized_audio_config()
            vad_config = self._optimizer.get_optimized_vad_config()
            transcriber_config = self._optimizer.get_optimized_transcriber_config()

            # Initialize audio capture with optimized settings and filtering
            self._audio_capture = AudioCapture(
                sample_rate=audio_config["sample_rate"],
                chunk_size=audio_config["chunk_size"],
                enable_filtering=self._enable_filtering,
            )
            logger.debug(
                f"Audio capture: sample_rate={audio_config['sample_rate']}, "
                f"chunk_size={audio_config['chunk_size']}, "
                f"filtering={'enabled' if self._enable_filtering else 'disabled'}"
            )

            # Initialize audio filtering pipeline if enabled
            if self._enable_filtering:
                try:
                    from .audio_filtering.audio_filter_pipeline import AudioFilterPipeline

                    self._audio_filter_pipeline = AudioFilterPipeline(
                        sample_rate=audio_config["sample_rate"],
                        enable_filtering=True,
                    )
                    # Set the filter pipeline in audio capture
                    self._audio_capture.set_audio_filter(self._audio_filter_pipeline)
                    logger.debug("Audio filtering pipeline initialized")
                except ImportError as e:
                    logger.warning(f"Audio filtering not available: {e}")
                    self._enable_filtering = False
                    self._audio_filter_pipeline = None
                except Exception as e:
                    logger.error(f"Failed to initialize audio filtering: {e}")
                    self._enable_filtering = False
                    self._audio_filter_pipeline = None

            # Initialize voice activity detector with optimized settings
            self._vad = VoiceActivityDetector(
                sample_rate=vad_config["sample_rate"],
                frame_duration=vad_config["frame_duration"],
            )

            # Initialize audio debugger if enabled
            if self._enable_audio_debugging:
                try:
                    from .audio_debugger import AudioDebugger

                    self._audio_debugger = AudioDebugger(
                        enabled=True,
                        output_dir=self._audio_debug_dir,
                    )
                    logger.debug("Audio debugger initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize audio debugger: {e}")
                    self._audio_debugger = None
            else:
                self._audio_debugger = None

            # Initialize transcriber with optimized settings
            self._transcriber = WhisperTranscriber(
                model_size=transcriber_config["model_size"],
                device=transcriber_config["device"],
                compute_type=transcriber_config["compute_type"],
                audio_debugger=self._audio_debugger,
            )

            # Check if transcriber model is available
            if not self._transcriber.is_model_available():
                logger.error("Whisper model not available")
                return False

            logger.debug(
                f"Speech-to-text initialized ({self._optimization_target} optimization)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def _cleanup_components(self) -> None:
        """Clean up all components."""
        try:
            if self._audio_capture and self._audio_capture.is_capturing():
                self._audio_capture.stop_capture()
        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")

        self._audio_capture = None
        self._vad = None
        self._transcriber = None

    async def start_listening(self) -> None:
        """Start the speech-to-text listening service."""
        if self._listening:
            logger.warning("Service is already listening")
            return

        try:
            # Initialize components if not already done
            if not self._initialize_components():
                raise RuntimeError("Failed to initialize speech-to-text components")

            # Start audio capture
            if self._audio_capture:
                logger.debug("🎤 Starting audio capture...")
                self._audio_capture.start_capture()

                # Log audio capture debug stats if available
                if hasattr(self._audio_capture, "get_debug_stats"):
                    debug_stats = self._audio_capture.get_debug_stats()
                    logger.debug(f"🎤 Audio capture status: {debug_stats}")
                else:
                    logger.debug("🎤 Audio capture started (debug stats not available)")

            self._listening = True

            # Start the real-time processing pipeline
            self._processing_task = asyncio.create_task(self._process_audio_pipeline())

            logger.debug("Speech-to-text service started listening")

        except AudioCaptureError as e:
            logger.error(f"Audio capture error: {e}")
            self._cleanup_components()
            raise
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
            self._cleanup_components()
            raise

    async def stop_listening(self) -> None:
        """Stop the speech-to-text listening service."""
        if not self._listening:
            return

        self._listening = False

        # Cancel processing task if running
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Clean up components
        self._cleanup_components()

        logger.debug("Speech-to-text service stopped listening")

    def get_latest_transcription_result(self) -> TranscriptionResult | None:
        """
        Get the most recent transcription result with confidence information.

        Returns:
            Latest TranscriptionResult or None if no transcription available
        """
        return self._latest_transcription_result

    def set_transcription_result_callback(
        self, callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """
        Set callback for real-time transcription results with confidence information.

        Args:
            callback: Function to call with TranscriptionResult objects
        """
        self._transcription_result_callback = callback

    def set_pipeline_callback(
        self, callback: Callable[[list[ProcessingResult]], None]
    ) -> None:
        """
        Set callback function for pipeline processing results.

        Args:
            callback: Function to call with pipeline processing results
        """
        self._pipeline_callback = callback

    def register_processing_handler(self, handler: ProcessingHandler) -> bool:
        """
        Register a processing handler for future system integration.

        Args:
            handler: Processing handler to register

        Returns:
            True if registration was successful
        """
        return self._processing_pipeline.register_handler(handler)

    def unregister_processing_handler(self, stage_name: str, handler_name: str) -> bool:
        """
        Unregister a processing handler.

        Args:
            stage_name: Name of the processing stage
            handler_name: Name of the handler to unregister

        Returns:
            True if unregistration was successful
        """
        from .interfaces import ProcessingStage

        # Convert stage name to ProcessingStage enum
        try:
            stage = ProcessingStage(stage_name)
            return self._processing_pipeline.unregister_handler(stage, handler_name)
        except ValueError:
            logger.error(f"Invalid processing stage: {stage_name}")
            return False

    def get_registered_handlers(self) -> dict[str, list[str]]:
        """
        Get information about registered processing handlers.

        Returns:
            Dictionary mapping stage names to handler names
        """
        return self._processing_pipeline.get_handler_info()

    def get_pipeline_stats(self) -> dict[str, Any]:
        """
        Get pipeline processing statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        return self._processing_pipeline.get_pipeline_stats()

    def _update_transcription(
        self, text: str, transcription_metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Update the latest transcription and notify callback.

        Args:
            text: New transcription text
            transcription_metadata: Optional metadata about the transcription
        """
        if not text or not text.strip():
            return

        # Trigger pipeline processing for future systems
        if transcription_metadata:
            asyncio.create_task(
                self._trigger_pipeline_processing(text, transcription_metadata)
            )

    def _update_transcription_with_result(
        self,
        transcription_result: TranscriptionResult,
        transcription_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update the latest transcription with full result information and notify callbacks.

        Args:
            transcription_result: TranscriptionResult object with confidence information
            transcription_metadata: Optional metadata about the transcription
        """
        if not transcription_result.text or not transcription_result.text.strip():
            return

        self._latest_transcription_result = transcription_result

        # Call transcription result callback if set (with confidence information)
        if self._transcription_result_callback:
            try:
                self._transcription_result_callback(transcription_result)
            except Exception as e:
                logger.error(f"Error in transcription result callback: {e}")

        # Trigger task detection if enabled
        if self._enable_task_detection and self._task_detection_service:
            asyncio.create_task(
                self._detect_task_from_transcription(transcription_result.text)
            )

        # Trigger pipeline processing for future systems
        if transcription_metadata:
            asyncio.create_task(
                self._trigger_pipeline_processing(
                    transcription_result.text, transcription_metadata
                )
            )

    async def _detect_task_from_transcription(self, text: str) -> None:
        """
        Detect and create tasks from transcribed text.

        Args:
            text: Transcribed text to analyze for tasks
        """
        try:
            logger.info(f"🎯 Starting task detection from transcription: '{text}'")
            result = await self._task_detection_service.detect_task_from_text(text)

            if result.task_detected and result.task:
                logger.info(
                    f"✅ Task detected from speech: '{result.task.description}' "
                    f"(confidence: {result.confidence:.2f}, "
                    f"priority: {result.task.priority}, "
                    f"due_date: {result.task.due_date})"
                )
            else:
                logger.info(
                    f"❌ No task detected from speech "
                    f"(confidence: {result.confidence:.2f})"
                )
                if result.error:
                    logger.warning(f"⚠️ Task detection error: {result.error}")

        except Exception as e:
            logger.error(
                f"❌ Error detecting task from transcription: {e}", exc_info=True
            )

    async def _trigger_pipeline_processing(
        self, text: str, metadata: dict[str, Any]
    ) -> None:
        """
        Trigger processing pipeline for downstream systems.

        Args:
            text: Transcribed text
            metadata: Transcription metadata
        """
        try:
            # Create processing context with all necessary metadata
            context = create_processing_context(
                text=text,
                confidence=metadata.get("confidence", 0.0),
                timestamp=metadata.get("timestamp", time.time()),
                processing_time=metadata.get("processing_time", 0.0),
                audio_duration=metadata.get("audio_duration", 0.0),
                sample_rate=metadata.get("sample_rate", DEFAULT_SAMPLE_RATE),
                chunk_count=metadata.get("chunk_count", 0),
                session_id=metadata.get("session_id"),
                user_id=metadata.get("user_id"),
                metadata=metadata.get("additional_metadata", {}),
            )

            # Process through pipeline
            results = await self._processing_pipeline.process_transcription(context)

            # Call pipeline callback if set
            if self._pipeline_callback:
                try:
                    self._pipeline_callback(results)
                except Exception as e:
                    logger.error(f"Error in pipeline callback: {e}")

            logger.debug(f"Pipeline processing completed with {len(results)} results")

        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")

    def is_listening(self) -> bool:
        """
        Check if the service is currently listening.

        Returns:
            True if listening, False otherwise
        """
        return self._listening

    def get_component_status(self) -> dict[str, bool]:
        """
        Get status of all components.

        Returns:
            Dictionary with component status information
        """
        return {
            "audio_capture": self._audio_capture is not None
            and self._audio_capture.is_capturing(),
            "vad": self._vad is not None,
            "transcriber": self._transcriber is not None
            and self._transcriber.is_model_available(),
            "listening": self._listening,
            "filtering": self.is_filtering_enabled(),
        }

    def get_performance_stats(self, time_window: float | None = None) -> dict:
        """
        Get performance statistics for the service.

        Args:
            time_window: Time window in seconds (None for all time)

        Returns:
            Dictionary with performance statistics
        """
        if not self._monitoring_enabled or not self._performance_monitor:
            return {"monitoring_disabled": True}
        return self._performance_monitor.get_overall_stats(time_window)

    def get_performance_report(self) -> str:
        """
        Get a human-readable performance report.

        Returns:
            Formatted performance report
        """
        if not self._monitoring_enabled or not self._performance_monitor:
            return "Performance monitoring is disabled"
        return self._performance_monitor.get_performance_report()

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics."""
        if self._monitoring_enabled and self._performance_monitor:
            self._performance_monitor.reset_metrics()

    def set_noise_profile(self, noise_sample: bytes) -> None:
        """
        Set noise profile for adaptive filtering.

        Args:
            noise_sample: Audio sample containing noise to profile
        """
        if self._audio_filter_pipeline:
            self._audio_filter_pipeline.set_noise_profile(noise_sample)
            logger.debug("Noise profile updated")
        else:
            logger.warning("Audio filtering not available for noise profile setting")

    def get_filter_stats(self) -> dict[str, Any]:
        """
        Get audio filtering statistics.

        Returns:
            Dictionary with filtering performance statistics
        """
        if self._audio_filter_pipeline:
            return self._audio_filter_pipeline.get_filter_stats()
        return {"filtering_disabled": True}

    def reset_adaptive_filters(self) -> None:
        """Reset adaptive filtering parameters."""
        if self._audio_filter_pipeline:
            self._audio_filter_pipeline.reset_adaptive_filters()
            logger.debug("Adaptive filters reset")
        else:
            logger.warning("Audio filtering not available for filter reset")

    def is_filtering_enabled(self) -> bool:
        """
        Check if audio filtering is enabled and available.

        Returns:
            True if filtering is enabled and available, False otherwise
        """
        return self._enable_filtering and self._audio_filter_pipeline is not None

    async def _process_audio_pipeline(self) -> None:
        """
        Main audio processing pipeline that runs continuously.

        This method:
        1. Continuously captures audio chunks
        2. Uses VAD to detect speech segments
        3. Transcribes speech segments using Whisper
        4. Delivers transcription results via callback
        """
        if not self._audio_capture or not self._vad or not self._transcriber:
            logger.error("Components not initialized for audio processing")
            return

        # Get optimized pipeline configuration
        pipeline_config = self._optimizer.get_optimized_pipeline_config()

        audio_buffer = []
        speech_buffer = []
        silence_counter = 0

        # Calculate optimized parameters
        frame_duration_ms = pipeline_config.get(
            "vad_frame_duration", 30
        )  # VAD frame duration
        max_silence_chunks = int(
            pipeline_config["max_silence_duration"] * 1000 / frame_duration_ms
        )
        max_buffer_size = int(
            pipeline_config["max_audio_buffer_size"] * 1000 / frame_duration_ms
        )
        max_speech_buffer = int(
            pipeline_config["max_audio_buffer_size"] * 1000 / frame_duration_ms
        )
        sample_rate = pipeline_config.get("sample_rate", DEFAULT_SAMPLE_RATE)
        int(
            pipeline_config["min_speech_duration"] * sample_rate * 2
        )  # bytes for min duration
        processing_interval = pipeline_config["processing_interval"]

        logger.debug("Starting real-time audio processing pipeline")

        try:
            while self._listening:
                try:
                    # Get audio chunk from capture (now async for filtering support)
                    audio_chunk = await self._audio_capture.get_audio_chunk()

                    if audio_chunk is None:
                        # No audio available, wait a bit
                        await asyncio.sleep(0.01)
                        continue

                    # Add to buffer for processing
                    audio_buffer.append(audio_chunk)

                    # Keep buffer size manageable using optimized size
                    if len(audio_buffer) > max_buffer_size:
                        audio_buffer.pop(0)

                    # Check for speech in current chunk with performance monitoring
                    # Process ALL VAD frames in the chunk to avoid missing speech
                    vad_frame_size = (
                        self._vad.frame_size * 2
                    )  # Convert samples to bytes (16-bit audio)

                    is_speech = False

                    if len(audio_chunk) >= vad_frame_size:
                        # Process all complete VAD frames in the chunk
                        frames_checked = 0
                        for i in range(0, len(audio_chunk), vad_frame_size):
                            frame = audio_chunk[i : i + vad_frame_size]

                            # Handle incomplete frames based on configuration
                            if len(frame) < vad_frame_size:
                                if config.VAD_PAD_INCOMPLETE_FRAMES:
                                    # Pad incomplete frame with zeros
                                    padding_size = vad_frame_size - len(frame)
                                    frame = frame + b"\x00" * padding_size
                                    logger.trace(
                                        f"Padded incomplete frame: "
                                        f"{len(frame) - padding_size} -> {len(frame)} bytes"
                                    )
                                else:
                                    # Skip incomplete frame when padding is disabled
                                    logger.trace(
                                        f"Skipping incomplete frame: "
                                        f"{len(frame)} < {vad_frame_size}"
                                    )
                                    continue

                            # Process frame (complete or padded)
                            if len(frame) == vad_frame_size:
                                frames_checked += 1
                                if self._monitoring_enabled:
                                    with PerformanceContext(
                                        "vad", metadata={"chunk_size": len(frame)}
                                    ):
                                        frame_has_speech = self._vad.is_speech(frame)
                                else:
                                    frame_has_speech = self._vad.is_speech(frame)

                                if frame_has_speech:
                                    is_speech = True
                                    break  # Found speech, no need to check remaining frames

                        if frames_checked > 1:
                            logger.trace(
                                f"Checked {frames_checked} VAD frames in chunk, "
                                f"speech detected: {is_speech}"
                            )
                    # Chunk is smaller than one VAD frame
                    elif config.VAD_PAD_INCOMPLETE_FRAMES:
                        # Pad the chunk to make it a complete frame
                        padding_size = vad_frame_size - len(audio_chunk)
                        padded_chunk = audio_chunk + b"\x00" * padding_size
                        logger.trace(
                            f"Padded small chunk: "
                            f"{len(audio_chunk)} -> {len(padded_chunk)} bytes"
                        )

                        # Process the padded frame
                        if self._monitoring_enabled:
                            with PerformanceContext(
                                "vad", metadata={"chunk_size": len(padded_chunk)}
                            ):
                                is_speech = self._vad.is_speech(padded_chunk)
                        else:
                            is_speech = self._vad.is_speech(padded_chunk)
                    else:
                        # Skip small chunks when padding is disabled
                        is_speech = False
                        logger.trace(
                            f"Skipping small chunk: {len(audio_chunk)} < {vad_frame_size}"
                        )

                    if is_speech:
                        # Add to speech buffer
                        speech_buffer.append(audio_chunk)
                        silence_counter = 0
                        logger.trace(
                            f"🗣️ Speech detected, buffer size: {len(speech_buffer)} chunks"
                        )
                    else:
                        silence_counter += 1

                        # If we have speech buffer and enough silence, process it
                        if speech_buffer and silence_counter >= max_silence_chunks:
                            logger.debug(
                                f"🔄 Processing segment: {len(speech_buffer)} chunks, "
                                f"silence_counter: {silence_counter}"
                            )
                            await self._process_speech_segment(speech_buffer)
                            speech_buffer = []
                            silence_counter = 0

                    # Prevent buffer from growing too large using optimized size
                    if len(speech_buffer) > max_speech_buffer:
                        # Process current buffer and start fresh
                        logger.debug(
                            f"🔄 Processing buffer: {len(speech_buffer)} chunks (max)"
                        )
                        await self._process_speech_segment(speech_buffer)
                        speech_buffer = []
                        silence_counter = 0

                    # Optimized delay to balance CPU usage and responsiveness
                    await asyncio.sleep(processing_interval)

                except asyncio.CancelledError:
                    logger.debug("Audio processing pipeline cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in audio processing pipeline: {e}")
                    # Continue processing despite errors
                    await asyncio.sleep(ERROR_RECOVERY_SLEEP)

            # Process any remaining speech buffer
            if speech_buffer:
                await self._process_speech_segment(speech_buffer)

        except Exception as e:
            logger.error(f"Fatal error in audio processing pipeline: {e}")
        finally:
            logger.debug("Audio processing pipeline stopped")

    async def _process_speech_segment(self, speech_chunks: list[bytes]) -> None:
        """
        Process a speech segment for transcription.

        Args:
            speech_chunks: List of audio chunks containing speech
        """
        if not speech_chunks or not self._transcriber:
            return

        try:
            # Combine speech chunks into single audio data
            combined_audio = b"".join(speech_chunks)

            # Get optimized minimum audio size
            pipeline_config = self._optimizer.get_optimized_pipeline_config()
            sample_rate = pipeline_config.get("sample_rate", DEFAULT_SAMPLE_RATE)
            min_audio_size = int(
                pipeline_config["min_speech_duration"] * sample_rate * 2
            )  # bytes for min duration

            # Skip very short segments using optimized minimum duration
            if len(combined_audio) < min_audio_size:
                logger.trace(f"Segment too short ({len(combined_audio)} bytes), skipping")
                return

            logger.trace(f"Processing speech segment of {len(combined_audio)} bytes")

            # Transcribe the audio segment with optional performance monitoring
            if self._monitoring_enabled:
                with PerformanceContext(
                    "transcription",
                    metadata={
                        "audio_size": len(combined_audio),
                        "chunk_count": len(speech_chunks),
                    },
                ) as ctx:
                    transcription_result = (
                        await self._transcriber.transcribe_audio_with_result(
                            combined_audio, source_sample_rate=sample_rate
                        )
                    )
                    ctx.set_metadata(
                        "transcription_length",
                        len(transcription_result.text)
                        if transcription_result.text
                        else 0,
                    )
            else:
                transcription_result = (
                    await self._transcriber.transcribe_audio_with_result(
                        combined_audio, source_sample_rate=sample_rate
                    )
                )

            if (
                transcription_result
                and transcription_result.text
                and transcription_result.text.strip()
            ):
                logger.debug(
                    f"Result: {transcription_result.text} "
                    f"({transcription_result.confidence:.1f})"
                )

                # Create metadata for pipeline processing
                transcription_metadata = {
                    "confidence": transcription_result.confidence,
                    "timestamp": transcription_result.timestamp,
                    "processing_time": transcription_result.processing_time,
                    "audio_duration": len(combined_audio)
                    / (sample_rate * 2),  # bytes to seconds (16-bit audio)
                    "sample_rate": sample_rate,
                    "chunk_count": len(speech_chunks),
                    "session_id": getattr(self, "_session_id", None),
                    "user_id": getattr(self, "_user_id", None),
                    "additional_metadata": {
                        "audio_size_bytes": len(combined_audio),
                        "optimization_target": self._optimization_target,
                        "monitoring_enabled": self._monitoring_enabled,
                    },
                }

                self._update_transcription_with_result(
                    transcription_result, transcription_metadata
                )
            else:
                logger.trace("Empty transcription result")

        except asyncio.CancelledError:
            logger.trace("Speech segment processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            # Don't re-raise to keep pipeline running
