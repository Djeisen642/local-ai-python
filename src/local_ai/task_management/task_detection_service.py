"""Task Detection Service for detecting and managing tasks from text input."""

import logging
import time

from .config import DEFAULT_CONFIDENCE_THRESHOLD
from .llm_classifier import LLMClassifier
from .models import TaskDetectionResult, TaskPriority
from .task_list_manager import TaskListManager

logger = logging.getLogger(__name__)


class TaskDetectionService:
    """
    Service for detecting tasks from text input.

    Coordinates LLM classification and task management to automatically
    detect and store tasks from text input sources.
    """

    def __init__(
        self,
        llm_classifier: LLMClassifier,
        task_manager: TaskListManager,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialize Task Detection Service.

        Args:
            llm_classifier: LLM classifier for task classification
            task_manager: Task list manager for task storage
            confidence_threshold: Minimum confidence for task creation
        """
        self._llm_classifier = llm_classifier
        self._task_manager = task_manager
        self._confidence_threshold = confidence_threshold

        logger.info(
            f"Task Detection Service initialized with confidence threshold: "
            f"{confidence_threshold}"
        )

    async def detect_task_from_text(self, text: str) -> TaskDetectionResult:
        """
        Detect and create task from text input.

        Args:
            text: Input text to analyze

        Returns:
            TaskDetectionResult with detection outcome
        """
        start_time = time.time()

        logger.info(f"🎯 Task detection started for text: '{text}'")

        # Validate input
        if not text or not text.strip():
            logger.warning("❌ Empty or whitespace-only text provided")
            return TaskDetectionResult(
                task_detected=False,
                task=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error="Empty or whitespace-only text provided",
            )

        try:
            # Classify text using LLM
            classification = await self._llm_classifier.classify_text(text)

            logger.info(
                f"📊 Classification result: is_task={classification.is_task}, "
                f"confidence={classification.confidence:.2f}, "
                f"description='{classification.description}', "
                f"priority={classification.priority}, due_date={classification.due_date}"
            )

            # Check if it's a task and meets confidence threshold
            if not classification.is_task:
                processing_time = time.time() - start_time
                logger.info(
                    f"❌ Not classified as a task (confidence: {classification.confidence:.2f})"
                )
                return TaskDetectionResult(
                    task_detected=False,
                    task=None,
                    confidence=classification.confidence,
                    processing_time=processing_time,
                )

            if classification.confidence < self._confidence_threshold:
                logger.info(
                    f"⚠️ Task confidence {classification.confidence:.2f} below threshold "
                    f"{self._confidence_threshold:.2f}, not creating task"
                )
                processing_time = time.time() - start_time
                return TaskDetectionResult(
                    task_detected=False,
                    task=None,
                    confidence=classification.confidence,
                    processing_time=processing_time,
                )

            # Extract task details
            description = classification.description or text
            priority = classification.priority or TaskPriority.MEDIUM
            due_date = classification.due_date
            metadata = classification.metadata.copy() if classification.metadata else {}

            logger.info(
                f"✅ Creating task: description='{description}', "
                f"priority={priority}, due_date={due_date}"
            )

            # Create task via Task List Manager
            task_id = await self._task_manager.add_task(
                description=description,
                priority=priority,
                source="text",
                confidence=classification.confidence,
                due_date=due_date,
                metadata=metadata,
            )

            # Retrieve created task
            task = await self._task_manager.get_task(task_id)

            processing_time = time.time() - start_time

            logger.info(
                f"🎉 Task created successfully: ID={task_id}, "
                f"description='{task.description}', "
                f"confidence={classification.confidence:.2f}, "
                f"processing_time={processing_time:.3f}s"
            )

            return TaskDetectionResult(
                task_detected=True,
                task=task,
                confidence=classification.confidence,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Task detection failed: {str(e)}"
            logger.error(error_msg)

            return TaskDetectionResult(
                task_detected=False,
                task=None,
                confidence=0.0,
                processing_time=processing_time,
                error=error_msg,
            )
