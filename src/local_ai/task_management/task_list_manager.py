"""Task List Manager for managing tasks with database persistence."""

import logging
import uuid
from datetime import datetime
from typing import Any

from .database import TaskDatabase
from .models import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class TaskListManager:
    """
    Manages task list with database persistence.

    Provides CRUD operations, statistics calculation, and history tracking
    for tasks. Maintains in-memory cache for performance and resilience.
    """

    def __init__(self, database: TaskDatabase) -> None:
        """
        Initialize Task List Manager.

        Args:
            database: Database instance for task persistence
        """
        self._database = database
        self._tasks_cache: dict[uuid.UUID, Task] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the manager and load existing tasks.

        Initializes database connection and loads all existing tasks
        into memory for fast access and statistics calculation.
        """
        logger.info("Initializing Task List Manager")

        # Initialize database
        await self._database.initialize()

        # Load existing tasks into cache
        tasks = await self._database.list_tasks()
        self._tasks_cache = {task.id: task for task in tasks}

        self._initialized = True
        logger.info(f"Task List Manager initialized with {len(self._tasks_cache)} tasks")

    async def add_task(
        self,
        description: str,
        priority: TaskPriority,
        source: str,
        confidence: float,
        due_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """
        Add a new task.

        Args:
            description: Task description
            priority: Task priority
            source: Source of task (e.g., 'voice', 'cli', 'mcp')
            confidence: Classification confidence score
            due_date: Optional due date
            metadata: Optional metadata dictionary

        Returns:
            UUID of created task

        Raises:
            DatabaseError: If task creation fails
        """
        task_id = uuid.uuid4()
        now = datetime.now()

        task = Task(
            id=task_id,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=now,
            updated_at=now,
            source=source,
            confidence=confidence,
            due_date=due_date,
            metadata=metadata or {},
        )

        # Persist to database
        await self._database.insert_task(task)

        # Update cache
        self._tasks_cache[task_id] = task

        logger.info(f"Added task {task_id}: {description}")
        return task_id

    async def get_task(self, task_id: uuid.UUID) -> Task:
        """
        Get a task by ID.

        Args:
            task_id: Task UUID

        Returns:
            Task object

        Raises:
            TaskNotFoundError: If task not found
        """
        # Try cache first
        if task_id in self._tasks_cache:
            return self._tasks_cache[task_id]

        # Fall back to database
        task = await self._database.get_task(task_id)
        self._tasks_cache[task_id] = task
        return task

    async def list_tasks(
        self,
        status: TaskStatus | None = None,
        priority: TaskPriority | None = None,
    ) -> list[Task]:
        """
        List tasks with optional filters.

        Args:
            status: Filter by status
            priority: Filter by priority

        Returns:
            List of tasks matching filters
        """
        tasks = await self._database.list_tasks(status=status, priority=priority)

        # Update cache with fetched tasks
        for task in tasks:
            self._tasks_cache[task.id] = task

        return tasks

    async def update_task_status(self, task_id: uuid.UUID, status: TaskStatus) -> None:
        """
        Update task status.

        Args:
            task_id: Task UUID
            status: New status

        Raises:
            TaskNotFoundError: If task not found
            DatabaseError: If update fails
        """
        await self._database.update_task_status(task_id, status)

        # Update cache
        if task_id in self._tasks_cache:
            self._tasks_cache[task_id].status = status
            self._tasks_cache[task_id].updated_at = datetime.now()
            if status == TaskStatus.COMPLETED:
                self._tasks_cache[task_id].completed_at = datetime.now()

        logger.info(f"Updated task {task_id} status to {status.value}")

    async def update_task(self, task_id: uuid.UUID, updates: dict[str, Any]) -> None:
        """
        Update multiple task fields.

        Args:
            task_id: Task UUID
            updates: Dictionary of field names and values

        Raises:
            TaskNotFoundError: If task not found
            DatabaseError: If update fails
        """
        await self._database.update_task(task_id, updates)

        # Update cache
        if task_id in self._tasks_cache:
            task = self._tasks_cache[task_id]
            for field, value in updates.items():
                if hasattr(task, field):
                    setattr(task, field, value)
            task.updated_at = datetime.now()

        logger.info(f"Updated task {task_id} fields: {list(updates.keys())}")

    async def delete_task(self, task_id: uuid.UUID) -> None:
        """
        Delete a task.

        Args:
            task_id: Task UUID

        Raises:
            TaskNotFoundError: If task not found
            DatabaseError: If deletion fails
        """
        await self._database.delete_task(task_id)

        # Remove from cache
        if task_id in self._tasks_cache:
            del self._tasks_cache[task_id]

        logger.info(f"Deleted task {task_id}")

    async def get_task_history(self, task_id: uuid.UUID) -> list[dict[str, Any]]:
        """
        Get task history.

        Args:
            task_id: Task UUID

        Returns:
            List of history entries
        """
        return await self._database.get_task_history(task_id)

    async def get_statistics(self) -> dict[str, int]:
        """
        Get task statistics.

        Calculates counts by status from in-memory cache for performance.

        Returns:
            Dictionary with task counts:
            - total: Total number of tasks
            - pending: Number of pending tasks
            - in_progress: Number of in-progress tasks
            - completed: Number of completed tasks
            - cancelled: Number of cancelled tasks
        """
        stats = {
            "total": len(self._tasks_cache),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
        }

        for task in self._tasks_cache.values():
            if task.status == TaskStatus.PENDING:
                stats["pending"] += 1
            elif task.status == TaskStatus.IN_PROGRESS:
                stats["in_progress"] += 1
            elif task.status == TaskStatus.COMPLETED:
                stats["completed"] += 1
            elif task.status == TaskStatus.CANCELLED:
                stats["cancelled"] += 1

        return stats

    async def shutdown(self) -> None:
        """
        Shutdown the manager and close database connection.

        Handles errors gracefully to ensure cleanup completes.
        """
        logger.info("Shutting down Task List Manager")
        try:
            await self._database.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")

        self._tasks_cache.clear()
        self._initialized = False
