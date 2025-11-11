"""Unit tests for Task List Manager (TDD - RED phase)."""

import uuid
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_ai.task_management.exceptions import DatabaseError, TaskNotFoundError
from local_ai.task_management.models import Task, TaskPriority, TaskStatus


@pytest.fixture
def mock_database() -> AsyncMock:
    """Create a mock database for testing."""
    db = AsyncMock()
    db.initialize = AsyncMock()
    db.close = AsyncMock()
    db.insert_task = AsyncMock()
    db.get_task = AsyncMock()
    db.list_tasks = AsyncMock(return_value=[])
    db.update_task_status = AsyncMock()
    db.update_task = AsyncMock()
    db.delete_task = AsyncMock()
    db.get_task_history = AsyncMock(return_value=[])
    db.get_statistics = AsyncMock(
        return_value={
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
        }
    )
    return db


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        id=uuid.uuid4(),
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source="test",
        confidence=0.9,
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskListManagerInitialization:
    """Test Task List Manager initialization."""

    async def test_initialization_with_database(self, mock_database: Any) -> None:
        """Test manager initializes with database connection."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        mock_database.initialize.assert_called_once()

    async def test_initialization_loads_existing_tasks(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test manager loads existing tasks on startup."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.list_tasks.return_value = [sample_task]

        manager = TaskListManager(mock_database)
        await manager.initialize()

        mock_database.list_tasks.assert_called_once()
        stats = await manager.get_statistics()
        assert stats["total"] == 1

    async def test_initialization_handles_empty_database(
        self, mock_database: Any
    ) -> None:
        """Test manager handles empty database gracefully."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.list_tasks.return_value = []

        manager = TaskListManager(mock_database)
        await manager.initialize()

        stats = await manager.get_statistics()
        assert stats["total"] == 0

    async def test_initialization_calculates_statistics(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test manager calculates statistics on initialization."""
        from local_ai.task_management.task_list_manager import TaskListManager

        tasks = [
            Task(
                id=uuid.uuid4(),
                description="Task 1",
                status=TaskStatus.PENDING,
                priority=TaskPriority.HIGH,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.9,
            ),
            Task(
                id=uuid.uuid4(),
                description="Task 2",
                status=TaskStatus.COMPLETED,
                priority=TaskPriority.LOW,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.8,
            ),
        ]
        mock_database.list_tasks.return_value = tasks

        manager = TaskListManager(mock_database)
        await manager.initialize()

        stats = await manager.get_statistics()
        assert stats["total"] == 2
        assert stats["pending"] == 1
        assert stats["completed"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskListManagerCRUD:
    """Test Task List Manager CRUD operations."""

    async def test_add_task_generates_uuid(self, mock_database: Any) -> None:
        """Test adding task generates UUID automatically."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        task_id = await manager.add_task(
            description="New task",
            priority=TaskPriority.HIGH,
            source="test",
            confidence=0.9,
        )

        assert isinstance(task_id, uuid.UUID)
        mock_database.insert_task.assert_called_once()

    async def test_add_task_with_due_date(self, mock_database: Any) -> None:
        """Test adding task with due date."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        due_date = datetime.now() + timedelta(days=7)
        task_id = await manager.add_task(
            description="Task with deadline",
            priority=TaskPriority.HIGH,
            source="test",
            confidence=0.9,
            due_date=due_date,
        )

        assert task_id is not None
        call_args = mock_database.insert_task.call_args[0][0]
        assert call_args.due_date == due_date

    async def test_add_task_with_metadata(self, mock_database: Any) -> None:
        """Test adding task with metadata."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        metadata = {"category": "work", "tags": ["urgent", "important"]}
        task_id = await manager.add_task(
            description="Task with metadata",
            priority=TaskPriority.HIGH,
            source="test",
            confidence=0.9,
            metadata=metadata,
        )

        assert task_id is not None
        call_args = mock_database.insert_task.call_args[0][0]
        assert call_args.metadata == metadata

    async def test_get_task_by_id(self, mock_database: Any, sample_task: Any) -> None:
        """Test retrieving task by ID."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.get_task.return_value = sample_task

        manager = TaskListManager(mock_database)
        await manager.initialize()

        task = await manager.get_task(sample_task.id)

        assert task.id == sample_task.id
        assert task.description == sample_task.description
        mock_database.get_task.assert_called_once_with(sample_task.id)

    async def test_get_task_not_found(self, mock_database: Any) -> None:
        """Test getting non-existent task raises error."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.get_task.side_effect = TaskNotFoundError("Task not found")

        manager = TaskListManager(mock_database)
        await manager.initialize()

        with pytest.raises(TaskNotFoundError):
            await manager.get_task(uuid.uuid4())

    async def test_list_all_tasks(self, mock_database: Any, sample_task: Any) -> None:
        """Test listing all tasks."""
        from local_ai.task_management.task_list_manager import TaskListManager

        tasks = [sample_task]
        mock_database.list_tasks.return_value = tasks

        manager = TaskListManager(mock_database)
        await manager.initialize()

        result = await manager.list_tasks()

        assert len(result) == 1
        assert result[0].id == sample_task.id

    async def test_list_tasks_by_status(self, mock_database: Any) -> None:
        """Test listing tasks filtered by status."""
        from local_ai.task_management.task_list_manager import TaskListManager

        pending_task = Task(
            id=uuid.uuid4(),
            description="Pending task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )
        mock_database.list_tasks.return_value = [pending_task]

        manager = TaskListManager(mock_database)
        await manager.initialize()

        result = await manager.list_tasks(status=TaskStatus.PENDING)

        assert len(result) == 1
        assert result[0].status == TaskStatus.PENDING
        mock_database.list_tasks.assert_called_with(
            status=TaskStatus.PENDING, priority=None
        )

    async def test_list_tasks_by_priority(self, mock_database: Any) -> None:
        """Test listing tasks filtered by priority."""
        from local_ai.task_management.task_list_manager import TaskListManager

        high_priority_task = Task(
            id=uuid.uuid4(),
            description="High priority task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )
        mock_database.list_tasks.return_value = [high_priority_task]

        manager = TaskListManager(mock_database)
        await manager.initialize()

        result = await manager.list_tasks(priority=TaskPriority.HIGH)

        assert len(result) == 1
        assert result[0].priority == TaskPriority.HIGH
        mock_database.list_tasks.assert_called_with(
            status=None, priority=TaskPriority.HIGH
        )

    async def test_update_task_status(self, mock_database: Any, sample_task: Any) -> None:
        """Test updating task status."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.update_task_status(sample_task.id, TaskStatus.IN_PROGRESS)

        mock_database.update_task_status.assert_called_once_with(
            sample_task.id, TaskStatus.IN_PROGRESS
        )

    async def test_update_task_status_to_completed(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test updating task status to completed sets completed_at."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.update_task_status(sample_task.id, TaskStatus.COMPLETED)

        mock_database.update_task_status.assert_called_once_with(
            sample_task.id, TaskStatus.COMPLETED
        )

    async def test_update_task_priority(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test updating task priority."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.update_task(sample_task.id, {"priority": TaskPriority.HIGH})

        mock_database.update_task.assert_called_once_with(
            sample_task.id, {"priority": TaskPriority.HIGH}
        )

    async def test_update_task_multiple_fields(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test updating multiple task fields."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        updates = {
            "priority": TaskPriority.HIGH,
            "due_date": datetime.now() + timedelta(days=3),
        }
        await manager.update_task(sample_task.id, updates)

        mock_database.update_task.assert_called_once_with(sample_task.id, updates)

    async def test_delete_task(self, mock_database: Any, sample_task: Any) -> None:
        """Test deleting task."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.delete_task(sample_task.id)

        mock_database.delete_task.assert_called_once_with(sample_task.id)

    async def test_delete_nonexistent_task(self, mock_database: Any) -> None:
        """Test deleting non-existent task raises error."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.delete_task.side_effect = TaskNotFoundError("Task not found")

        manager = TaskListManager(mock_database)
        await manager.initialize()

        with pytest.raises(TaskNotFoundError):
            await manager.delete_task(uuid.uuid4())


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskListManagerStatistics:
    """Test Task List Manager statistics calculation."""

    async def test_get_statistics_empty(self, mock_database: Any) -> None:
        """Test statistics for empty task list."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.list_tasks.return_value = []

        manager = TaskListManager(mock_database)
        await manager.initialize()

        stats = await manager.get_statistics()

        assert stats["total"] == 0
        assert stats["pending"] == 0
        assert stats["in_progress"] == 0
        assert stats["completed"] == 0
        assert stats["cancelled"] == 0

    async def test_get_statistics_with_tasks(self, mock_database: Any) -> None:
        """Test statistics calculation with multiple tasks."""
        from local_ai.task_management.task_list_manager import TaskListManager

        tasks = [
            Task(
                id=uuid.uuid4(),
                description="Task 1",
                status=TaskStatus.PENDING,
                priority=TaskPriority.HIGH,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.9,
            ),
            Task(
                id=uuid.uuid4(),
                description="Task 2",
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.8,
            ),
            Task(
                id=uuid.uuid4(),
                description="Task 3",
                status=TaskStatus.IN_PROGRESS,
                priority=TaskPriority.HIGH,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.95,
            ),
            Task(
                id=uuid.uuid4(),
                description="Task 4",
                status=TaskStatus.COMPLETED,
                priority=TaskPriority.LOW,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.85,
            ),
        ]
        mock_database.list_tasks.return_value = tasks

        manager = TaskListManager(mock_database)
        await manager.initialize()

        stats = await manager.get_statistics()

        assert stats["total"] == 4
        assert stats["pending"] == 2
        assert stats["in_progress"] == 1
        assert stats["completed"] == 1
        assert stats["cancelled"] == 0

    async def test_statistics_update_after_add(self, mock_database: Any) -> None:
        """Test statistics update after adding task."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        # Mock the database to return updated list
        new_task = Task(
            id=uuid.uuid4(),
            description="New task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )
        mock_database.list_tasks.return_value = [new_task]

        await manager.add_task(
            description="New task",
            priority=TaskPriority.MEDIUM,
            source="test",
            confidence=0.9,
        )

        # Refresh statistics
        stats = await manager.get_statistics()
        assert stats["total"] >= 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskListManagerHistory:
    """Test Task List Manager history tracking."""

    async def test_get_task_history(self, mock_database: Any, sample_task: Any) -> None:
        """Test retrieving task history."""
        from local_ai.task_management.task_list_manager import TaskListManager

        history = [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "created",
                "field_name": None,
                "old_value": None,
                "new_value": None,
                "source": "test",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "action": "status_updated",
                "field_name": "status",
                "old_value": "pending",
                "new_value": "in_progress",
                "source": "system",
            },
        ]
        mock_database.get_task_history.return_value = history

        manager = TaskListManager(mock_database)
        await manager.initialize()

        result = await manager.get_task_history(sample_task.id)

        assert len(result) == 2
        assert result[0]["action"] == "created"
        assert result[1]["action"] == "status_updated"
        mock_database.get_task_history.assert_called_once_with(sample_task.id)

    async def test_history_tracks_creation(self, mock_database: Any) -> None:
        """Test history tracks task creation."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.add_task(
            description="New task",
            priority=TaskPriority.HIGH,
            source="test",
            confidence=0.9,
        )

        # Verify insert_task was called (which creates history entry)
        mock_database.insert_task.assert_called_once()

    async def test_history_tracks_status_updates(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test history tracks status updates."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.update_task_status(sample_task.id, TaskStatus.COMPLETED)

        # Verify update_task_status was called (which creates history entry)
        mock_database.update_task_status.assert_called_once()

    async def test_history_tracks_deletions(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test history tracks task deletions."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()

        await manager.delete_task(sample_task.id)

        # Verify delete_task was called (which creates history entry)
        mock_database.delete_task.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskListManagerErrorHandling:
    """Test Task List Manager error handling."""

    async def test_database_error_on_add(self, mock_database: Any) -> None:
        """Test handling database error when adding task."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.insert_task.side_effect = DatabaseError("Database error")

        manager = TaskListManager(mock_database)
        await manager.initialize()

        with pytest.raises(DatabaseError):
            await manager.add_task(
                description="New task",
                priority=TaskPriority.HIGH,
                source="test",
                confidence=0.9,
            )

    async def test_database_error_on_update(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test handling database error when updating task."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.update_task_status.side_effect = DatabaseError("Database error")

        manager = TaskListManager(mock_database)
        await manager.initialize()

        with pytest.raises(DatabaseError):
            await manager.update_task_status(sample_task.id, TaskStatus.COMPLETED)

    async def test_database_error_on_delete(
        self, mock_database: Any, sample_task: Any
    ) -> None:
        """Test handling database error when deleting task."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.delete_task.side_effect = DatabaseError("Database error")

        manager = TaskListManager(mock_database)
        await manager.initialize()

        with pytest.raises(DatabaseError):
            await manager.delete_task(sample_task.id)

    async def test_maintains_in_memory_state_on_failure(self, mock_database: Any) -> None:
        """Test manager maintains in-memory state when database fails."""
        from local_ai.task_management.task_list_manager import TaskListManager

        # First call succeeds, second fails
        mock_database.insert_task.side_effect = [None, DatabaseError("Database error")]

        manager = TaskListManager(mock_database)
        await manager.initialize()

        # First add succeeds
        task_id = await manager.add_task(
            description="Task 1",
            priority=TaskPriority.HIGH,
            source="test",
            confidence=0.9,
        )
        assert task_id is not None

        # Second add fails but manager should still be operational
        with pytest.raises(DatabaseError):
            await manager.add_task(
                description="Task 2",
                priority=TaskPriority.HIGH,
                source="test",
                confidence=0.9,
            )

    async def test_graceful_shutdown(self, mock_database: Any) -> None:
        """Test manager shuts down gracefully."""
        from local_ai.task_management.task_list_manager import TaskListManager

        manager = TaskListManager(mock_database)
        await manager.initialize()
        await manager.shutdown()

        mock_database.close.assert_called_once()

    async def test_shutdown_handles_errors(self, mock_database: Any) -> None:
        """Test shutdown handles errors gracefully."""
        from local_ai.task_management.task_list_manager import TaskListManager

        mock_database.close.side_effect = DatabaseError("Close error")

        manager = TaskListManager(mock_database)
        await manager.initialize()

        # Should not raise exception
        await manager.shutdown()
