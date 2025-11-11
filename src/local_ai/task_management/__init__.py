"""Task management module for automatic task detection and management."""

from local_ai.task_management.models import (
    ClassificationResult,
    Task,
    TaskDetectionResult,
    TaskPriority,
    TaskStatus,
)

__all__ = [
    "Task",
    "TaskStatus",
    "TaskPriority",
    "ClassificationResult",
    "TaskDetectionResult",
]
