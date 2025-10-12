"""Main speech-to-text service orchestrator."""

import asyncio
import logging
from collections.abc import Callable
from typing import Optional

from .audio_capture import AudioCapture, AudioCaptureError
from .vad import VoiceActivityDetector
from .transcriber import WhisperTranscriber
from .optimization import get_optimizer
from .performance_monitor import get_performance_monitor, PerformanceContext

logger = logging.getLogger(__name__)


class SpeechToTextService:
    """Main orchestrator that coordinates all speech-to-text components."""

    def __init__(self, optimization_target: str = "balanced", enable_monitoring: bool = False, use_cache: bool = True) -> None:
        """
        Initialize the speech-to-text service.
        
        Args:
            optimization_target: "latency", "accuracy", "resource", or "balanced"
            enable_monitoring: Whether to enable performance monitoring (adds overhead)
            use_cache: Whether to use cached optimization data
        """
        self._listening = False
        self._transcription_callback: Callable[[str], None] | None = None
        self._latest_transcription: str | None = None
        
        # Initialize components
        self._audio_capture: Optional[AudioCapture] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._transcriber: Optional[WhisperTranscriber] = None
        
        # Processing state
        self._processing_task: Optional[asyncio.Task] = None
        
        # Performance optimization
        self._use_cache = use_cache
        self._optimizer = get_optimizer(use_cache=use_cache)
        self._optimization_target = optimization_target
        self._optimized_config = self._get_optimized_config()
        
        # Optional performance monitoring
        self._monitoring_enabled = enable_monitoring
        self._performance_monitor = get_performance_monitor() if enable_monitoring else None

    def _get_optimized_config(self) -> dict:
        """Get optimized configuration based on target."""
        from .optimization import get_optimized_config
        return get_optimized_config(self._optimization_target, use_cache=self._use_cache)

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
            
            # Initialize audio capture with optimized settings
            self._audio_capture = AudioCapture(
                sample_rate=audio_config["sample_rate"],
                chunk_size=audio_config["chunk_size"]
            )
            
            # Initialize voice activity detector with optimized settings
            self._vad = VoiceActivityDetector(
                sample_rate=vad_config["sample_rate"],
                frame_duration=vad_config["frame_duration"]
            )
            
            # Initialize transcriber with optimized settings
            self._transcriber = WhisperTranscriber(
                model_size=transcriber_config["model_size"]
            )
            
            # Check if transcriber model is available
            if not self._transcriber.is_model_available():
                logger.error("Whisper model not available")
                return False
            
            logger.info(f"All speech-to-text components initialized successfully with {self._optimization_target} optimization")
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
                self._audio_capture.start_capture()
            
            self._listening = True
            
            # Start the real-time processing pipeline
            self._processing_task = asyncio.create_task(self._process_audio_pipeline())
            
            logger.info("Speech-to-text service started listening")
            
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
        
        logger.info("Speech-to-text service stopped listening")

    def get_latest_transcription(self) -> str | None:
        """
        Get the most recent transcription result.

        Returns:
            Latest transcription text or None if no transcription available
        """
        return self._latest_transcription

    def set_transcription_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function for real-time transcription updates.

        Args:
            callback: Function to call with transcription results
        """
        self._transcription_callback = callback

    def _update_transcription(self, text: str) -> None:
        """
        Update the latest transcription and notify callback.
        
        Args:
            text: New transcription text
        """
        if not text or not text.strip():
            return
        
        self._latest_transcription = text
        
        # Call callback if set
        if self._transcription_callback:
            try:
                self._transcription_callback(text)
            except Exception as e:
                logger.error(f"Error in transcription callback: {e}")

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
            "audio_capture": self._audio_capture is not None and self._audio_capture.is_capturing(),
            "vad": self._vad is not None,
            "transcriber": self._transcriber is not None and self._transcriber.is_model_available(),
            "listening": self._listening
        }

    def get_performance_stats(self, time_window: Optional[float] = None) -> dict:
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
        frame_duration_ms = 30  # VAD frame duration
        max_silence_chunks = int(pipeline_config["max_silence_duration"] * 1000 / frame_duration_ms)
        max_buffer_size = int(pipeline_config["max_audio_buffer_size"] * 1000 / frame_duration_ms)
        max_speech_buffer = int(pipeline_config["max_audio_buffer_size"] * 1000 / frame_duration_ms)
        min_audio_size = int(pipeline_config["min_speech_duration"] * 16000 * 2)  # bytes for min duration
        processing_interval = pipeline_config["processing_interval"]
        
        logger.info("Starting real-time audio processing pipeline")
        
        try:
            while self._listening:
                try:
                    # Get audio chunk from capture
                    audio_chunk = self._audio_capture.get_audio_chunk()
                    
                    if audio_chunk is None:
                        # No audio available, wait a bit
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Add to buffer for processing
                    audio_buffer.append(audio_chunk)
                    
                    # Keep buffer size manageable using optimized size
                    if len(audio_buffer) > max_buffer_size:
                        audio_buffer.pop(0)
                    
                    # Check for speech in current chunk with optional performance monitoring
                    if self._monitoring_enabled:
                        with PerformanceContext("vad", metadata={"chunk_size": len(audio_chunk)}):
                            is_speech = self._vad.is_speech(audio_chunk)
                    else:
                        is_speech = self._vad.is_speech(audio_chunk)
                    
                    if is_speech:
                        # Add to speech buffer
                        speech_buffer.append(audio_chunk)
                        silence_counter = 0
                        logger.debug("Speech detected, adding to buffer")
                    else:
                        silence_counter += 1
                        
                        # If we have speech buffer and enough silence, process it
                        if speech_buffer and silence_counter >= max_silence_chunks:
                            await self._process_speech_segment(speech_buffer)
                            speech_buffer = []
                            silence_counter = 0
                    
                    # Prevent buffer from growing too large using optimized size
                    if len(speech_buffer) > max_speech_buffer:
                        # Process current buffer and start fresh
                        await self._process_speech_segment(speech_buffer)
                        speech_buffer = []
                        silence_counter = 0
                    
                    # Optimized delay to balance CPU usage and responsiveness
                    await asyncio.sleep(processing_interval)
                    
                except asyncio.CancelledError:
                    logger.info("Audio processing pipeline cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in audio processing pipeline: {e}")
                    # Continue processing despite errors
                    await asyncio.sleep(0.1)
            
            # Process any remaining speech buffer
            if speech_buffer:
                await self._process_speech_segment(speech_buffer)
                
        except Exception as e:
            logger.error(f"Fatal error in audio processing pipeline: {e}")
        finally:
            logger.info("Audio processing pipeline stopped")

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
            combined_audio = b''.join(speech_chunks)
            
            # Get optimized minimum audio size
            pipeline_config = self._optimizer.get_optimized_pipeline_config()
            min_audio_size = int(pipeline_config["min_speech_duration"] * 16000 * 2)  # bytes for min duration
            
            # Skip very short segments using optimized minimum duration
            if len(combined_audio) < min_audio_size:
                logger.debug(f"Speech segment too short ({len(combined_audio)} bytes), skipping transcription")
                return
            
            logger.debug(f"Processing speech segment of {len(combined_audio)} bytes")
            
            # Transcribe the audio segment with optional performance monitoring
            if self._monitoring_enabled:
                with PerformanceContext("transcription", metadata={"audio_size": len(combined_audio), "chunk_count": len(speech_chunks)}) as ctx:
                    transcription = await self._transcriber.transcribe_audio(combined_audio)
                    ctx.set_metadata("transcription_length", len(transcription) if transcription else 0)
            else:
                transcription = await self._transcriber.transcribe_audio(combined_audio)
            
            if transcription and transcription.strip():
                logger.info(f"Transcription result: {transcription}")
                self._update_transcription(transcription)
            else:
                logger.debug("Empty transcription result")
                
        except asyncio.CancelledError:
            logger.debug("Speech segment processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            # Don't re-raise to keep pipeline running
