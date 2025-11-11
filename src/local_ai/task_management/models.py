"""Data models for task management functionality."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """Represents a task item."""

    id: UUID
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    updated_at: datetime
    source: str
    confidence: float
    due_date: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Result of LLM task classification."""

    is_task: bool
    confidence: float
    description: str | None = None
    priority: TaskPriority | None = None
    due_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDetectionResult:
    """Result of task detection process."""

    task_detected: bool
    task: Task | None = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error: str | None = None
