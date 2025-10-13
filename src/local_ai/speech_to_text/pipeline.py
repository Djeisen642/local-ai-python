"""Plugin-style processing pipeline implementation."""

import asyncio
import logging
from .logging_utils import get_logger
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set
from datetime import datetime

from .interfaces import (
    ProcessingHandler,
    ProcessingPipeline,
    ProcessingContext,
    ProcessingResult,
    ProcessingStage
)

logger = get_logger(__name__)


class PluginProcessingPipeline(ProcessingPipeline):
    """Plugin-style processing pipeline for speech-to-text integration."""
    
    def __init__(self) -> None:
        """Initialize the processing pipeline."""
        self._handlers: Dict[ProcessingStage, Dict[str, ProcessingHandler]] = defaultdict(dict)
        self._processing_lock = asyncio.Lock()
        self._pipeline_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "average_processing_time": 0.0,
            "last_processed": None
        }
    
    def register_handler(self, handler: ProcessingHandler) -> bool:
        """
        Register a processing handler.
        
        Args:
            handler: Handler to register
            
        Returns:
            True if registration was successful
        """
        try:
            stage = handler.stage
            name = handler.name
            
            # Check if handler with same name already exists for this stage
            if name in self._handlers[stage]:
                logger.warning(f"Handler '{name}' already registered for stage '{stage.value}', replacing")
            
            self._handlers[stage][name] = handler
            logger.debug(f"Registered handler '{name}' for stage '{stage.value}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register handler: {e}")
            return False
    
    def unregister_handler(self, stage: ProcessingStage, name: str) -> bool:
        """
        Unregister a processing handler.
        
        Args:
            stage: Processing stage
            name: Handler name
            
        Returns:
            True if unregistration was successful
        """
        try:
            if stage in self._handlers and name in self._handlers[stage]:
                del self._handlers[stage][name]
                logger.debug(f"Unregistered handler '{name}' from stage '{stage.value}'")
                return True
            else:
                logger.warning(f"Handler '{name}' not found for stage '{stage.value}'")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister handler: {e}")
            return False
    
    def get_registered_handlers(self, stage: Optional[ProcessingStage] = None) -> List[ProcessingHandler]:
        """
        Get list of registered handlers.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of registered handlers
        """
        if stage is not None:
            return list(self._handlers[stage].values())
        
        # Return all handlers from all stages
        all_handlers = []
        for stage_handlers in self._handlers.values():
            all_handlers.extend(stage_handlers.values())
        return all_handlers
    
    async def process_transcription(self, context: ProcessingContext) -> List[ProcessingResult]:
        """
        Process transcription through the entire pipeline.
        
        Args:
            context: Processing context with transcription data
            
        Returns:
            List of results from each processing stage
        """
        async with self._processing_lock:
            start_time = datetime.now()
            results = []
            
            try:
                # Process stages in order: EMBEDDING -> RESPONSE_GENERATION -> TEXT_TO_SPEECH
                processing_order = [
                    ProcessingStage.EMBEDDING,
                    ProcessingStage.RESPONSE_GENERATION,
                    ProcessingStage.TEXT_TO_SPEECH
                ]
                
                for stage in processing_order:
                    stage_results = await self._process_stage(stage, context)
                    results.extend(stage_results)
                    
                    # If any critical stage fails, we might want to stop processing
                    # For now, we continue processing even if some stages fail
                
                # Update pipeline statistics
                self._update_stats(start_time, True)
                logger.debug(f"Successfully processed transcription through {len(results)} handlers")
                
            except Exception as e:
                logger.error(f"Error processing transcription through pipeline: {e}")
                self._update_stats(start_time, False)
                
                # Add error result
                error_result = ProcessingResult(
                    success=False,
                    stage=ProcessingStage.TRANSCRIPTION,
                    data=None,
                    error=str(e),
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
                results.append(error_result)
            
            return results
    
    async def _process_stage(self, stage: ProcessingStage, context: ProcessingContext) -> List[ProcessingResult]:
        """
        Process a specific stage with all registered handlers.
        
        Args:
            stage: Processing stage to execute
            context: Processing context
            
        Returns:
            List of results from stage handlers
        """
        results = []
        stage_handlers = self._handlers.get(stage, {})
        
        if not stage_handlers:
            logger.debug(f"No handlers registered for stage '{stage.value}'")
            return results
        
        # Process all handlers for this stage concurrently
        tasks = []
        for name, handler in stage_handlers.items():
            if handler.can_handle(context):
                task = asyncio.create_task(
                    self._execute_handler(handler, context),
                    name=f"{stage.value}_{name}"
                )
                tasks.append(task)
            else:
                logger.debug(f"Handler '{name}' cannot handle context for stage '{stage.value}'")
        
        if tasks:
            # Wait for all handlers to complete
            handler_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(handler_results):
                if isinstance(result, Exception):
                    # Handler raised an exception
                    error_result = ProcessingResult(
                        success=False,
                        stage=stage,
                        data=None,
                        error=str(result),
                        processing_time=0.0
                    )
                    results.append(error_result)
                    logger.error(f"Handler {tasks[i].get_name()} failed: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def _execute_handler(self, handler: ProcessingHandler, context: ProcessingContext) -> ProcessingResult:
        """
        Execute a single handler with timing and error handling.
        
        Args:
            handler: Handler to execute
            context: Processing context
            
        Returns:
            Processing result
        """
        start_time = datetime.now()
        
        try:
            # Update context stage
            context.stage = handler.stage
            
            # Execute handler
            result = await handler.process(context)
            
            # Add timing information if not already present
            if result.processing_time is None:
                result.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.debug(f"Handler '{handler.name}' completed in {result.processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Handler '{handler.name}' failed after {processing_time:.3f}s: {e}")
            
            return ProcessingResult(
                success=False,
                stage=handler.stage,
                data=None,
                error=str(e),
                processing_time=processing_time
            )
    
    def _update_stats(self, start_time: datetime, success: bool) -> None:
        """
        Update pipeline processing statistics.
        
        Args:
            start_time: Processing start time
            success: Whether processing was successful
        """
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self._pipeline_stats["total_processed"] += 1
        if success:
            self._pipeline_stats["successful_processed"] += 1
        else:
            self._pipeline_stats["failed_processed"] += 1
        
        # Update average processing time
        total = self._pipeline_stats["total_processed"]
        current_avg = self._pipeline_stats["average_processing_time"]
        self._pipeline_stats["average_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        self._pipeline_stats["last_processed"] = datetime.now().isoformat()
    
    def get_pipeline_stats(self) -> Dict:
        """
        Get pipeline processing statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        return self._pipeline_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._pipeline_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "average_processing_time": 0.0,
            "last_processed": None
        }
        logger.debug("Pipeline statistics reset")
    
    def get_handler_info(self) -> Dict[str, List[str]]:
        """
        Get information about registered handlers.
        
        Returns:
            Dictionary mapping stage names to handler names
        """
        info = {}
        for stage, handlers in self._handlers.items():
            info[stage.value] = list(handlers.keys())
        return info


def create_processing_context(
    text: str,
    confidence: float,
    timestamp: float,
    processing_time: float,
    audio_duration: float,
    sample_rate: int,
    chunk_count: int,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> ProcessingContext:
    """
    Create a processing context from transcription data.
    
    Args:
        text: Transcribed text
        confidence: Transcription confidence score
        timestamp: Transcription timestamp
        processing_time: Time taken for transcription
        audio_duration: Duration of audio in seconds
        sample_rate: Audio sample rate
        chunk_count: Number of audio chunks processed
        session_id: Optional session identifier
        user_id: Optional user identifier
        metadata: Optional additional metadata
        
    Returns:
        ProcessingContext instance
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    return ProcessingContext(
        text=text,
        confidence=confidence,
        timestamp=timestamp,
        processing_time=processing_time,
        audio_duration=audio_duration,
        sample_rate=sample_rate,
        chunk_count=chunk_count,
        stage=ProcessingStage.TRANSCRIPTION,
        session_id=session_id,
        user_id=user_id,
        metadata=metadata or {}
    )