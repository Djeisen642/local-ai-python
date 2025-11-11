"""Abstract interfaces for task management system."""

from abc import ABC, abstractmethod

from local_ai.task_management.models import ClassificationResult, TaskDetectionResult


class TaskDetectionService(ABC):
    """Abstract interface for task detection service."""

    @abstractmethod
    async def detect_task_from_text(self, text: str) -> TaskDetectionResult:
        """
        Detect and create a task from text input.

        This method analyzes the provided text to determine if it represents
        an actionable task. If a task is detected with sufficient confidence,
        it extracts task details and creates a task entry.

        Args:
            text: Input text to analyze for task detection

        Returns:
            TaskDetectionResult containing detection status, task (if created),
            classification details, and any errors

        Raises:
            TaskDetectionError: If task detection process fails
            ClassificationError: If LLM classification fails
            StorageError: If task storage fails
        """
        pass

    @abstractmethod
    async def classify_text(self, text: str) -> ClassificationResult:
        """
        Classify text to determine if it represents a task.

        This method uses the LLM classifier to analyze text and extract
        task-related information without creating a task entry.

        Args:
            text: Input text to classify

        Returns:
            ClassificationResult with task detection confidence and details

        Raises:
            ClassificationError: If classification process fails
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the task detection service.

        This method sets up all required components including LLM connection,
        database initialization, and configuration loading.

        Raises:
            LLMConnectionError: If unable to connect to LLM service
            StorageError: If unable to initialize storage
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the task detection service.

        This method performs cleanup including closing database connections
        and releasing resources.
        """
        pass
