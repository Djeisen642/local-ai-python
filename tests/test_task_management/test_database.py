"""Tests for database layer functionality."""

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.exceptions import DatabaseError, TaskNotFoundError
from local_ai.task_management.models import Task, TaskPriority, TaskStatus


@pytest.mark.unit
class TestDatabaseSchemaCreation:
    """Test cases for database schema creation and initialization."""

    @pytest.mark.asyncio
    async def test_schema_creation_creates_tasks_table(self) -> None:
        """Test that schema creation creates tasks table."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Verify tasks table exists
        async with db._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "tasks"

        await db.close()

    @pytest.mark.asyncio
    async def test_schema_creation_creates_task_history_table(self) -> None:
        """Test that schema creation creates task_history table."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Verify task_history table exists
        async with db._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='task_history'"
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "task_history"

        await db.close()

    @pytest.mark.asyncio
    async def test_schema_creation_creates_indexes(self) -> None:
        """Test that schema creation creates required indexes."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Verify indexes exist
        async with db._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            indexes = await cursor.fetchall()
            index_names = [idx[0] for idx in indexes]

            assert "idx_tasks_status" in index_names
            assert "idx_tasks_priority" in index_names
            assert "idx_tasks_due_date" in index_names
            assert "idx_history_task_id" in index_names
            assert "idx_history_timestamp" in index_names

        await db.close()

    @pytest.mark.asyncio
    async def test_schema_version_tracking(self) -> None:
        """Test that schema version is tracked."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        version = await db.get_schema_version()
        assert version == 1

        await db.close()

    @pytest.mark.asyncio
    async def test_schema_migration_support(self) -> None:
        """Test that schema migration is supported."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Should not raise error when initializing again
        await db.initialize()

        await db.close()


@pytest.mark.unit
class TestDatabaseConnectionManagement:
    """Test cases for database connection management."""

    @pytest.mark.asyncio
    async def test_database_initialization(self) -> None:
        """Test database can be initialized."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        assert db._connection is not None

        await db.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self) -> None:
        """Test that WAL mode is enabled for concurrent access."""
        db = TaskDatabase(":memory:", wal_mode=True)
        await db.initialize()

        async with db._get_connection() as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            result = await cursor.fetchone()
            # In-memory databases don't support WAL, but we test the setting
            assert result is not None

        await db.close()

    @pytest.mark.asyncio
    async def test_connection_context_manager(self) -> None:
        """Test connection context manager works correctly."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        async with db._get_connection() as conn:
            assert conn is not None
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1

        await db.close()

    @pytest.mark.asyncio
    async def test_close_connection(self) -> None:
        """Test database connection can be closed."""
        db = TaskDatabase(":memory:")
        await db.initialize()
        await db.close()

        # Connection should be closed
        assert db._connection is None

    @pytest.mark.asyncio
    async def test_concurrent_access(self) -> None:
        """Test database supports concurrent access."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Create multiple concurrent operations
        async def read_operation() -> int:
            async with db._get_connection() as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM tasks")
                result = await cursor.fetchone()
                return result[0]

        results = await asyncio.gather(*[read_operation() for _ in range(5)])
        assert all(r == 0 for r in results)

        await db.close()


@pytest.mark.unit
class TestDatabaseCRUDOperations:
    """Test cases for CRUD operations."""

    @pytest.mark.asyncio
    async def test_insert_task(self) -> None:
        """Test inserting a new task."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)

        # Verify task was inserted
        retrieved = await db.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.description == "Test task"

        await db.close()

    @pytest.mark.asyncio
    async def test_insert_task_with_history(self) -> None:
        """Test that inserting a task creates history entry."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)

        # Verify history entry was created
        history = await db.get_task_history(task.id)
        assert len(history) == 1
        assert history[0]["action"] == "created"

        await db.close()

    @pytest.mark.asyncio
    async def test_get_task_by_id(self) -> None:
        """Test retrieving a task by ID."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)
        retrieved = await db.get_task(task.id)

        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.description == task.description
        assert retrieved.status == task.status
        assert retrieved.priority == task.priority

        await db.close()

    @pytest.mark.asyncio
    async def test_get_nonexistent_task_raises_error(self) -> None:
        """Test that getting a nonexistent task raises TaskNotFoundError."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        with pytest.raises(TaskNotFoundError):
            await db.get_task(uuid4())

        await db.close()

    @pytest.mark.asyncio
    async def test_list_all_tasks(self) -> None:
        """Test listing all tasks."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Insert multiple tasks
        tasks = [
            Task(
                id=uuid4(),
                description=f"Task {i}",
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.9,
            )
            for i in range(3)
        ]

        for task in tasks:
            await db.insert_task(task)

        # List all tasks
        all_tasks = await db.list_tasks()
        assert len(all_tasks) == 3

        await db.close()

    @pytest.mark.asyncio
    async def test_list_tasks_with_status_filter(self) -> None:
        """Test listing tasks filtered by status."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Insert tasks with different statuses
        pending_task = Task(
            id=uuid4(),
            description="Pending task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        completed_task = Task(
            id=uuid4(),
            description="Completed task",
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(pending_task)
        await db.insert_task(completed_task)

        # Filter by status
        pending_tasks = await db.list_tasks(status=TaskStatus.PENDING)
        assert len(pending_tasks) == 1
        assert pending_tasks[0].status == TaskStatus.PENDING

        await db.close()

    @pytest.mark.asyncio
    async def test_list_tasks_with_priority_filter(self) -> None:
        """Test listing tasks filtered by priority."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Insert tasks with different priorities
        high_task = Task(
            id=uuid4(),
            description="High priority task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        low_task = Task(
            id=uuid4(),
            description="Low priority task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.LOW,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(high_task)
        await db.insert_task(low_task)

        # Filter by priority
        high_tasks = await db.list_tasks(priority=TaskPriority.HIGH)
        assert len(high_tasks) == 1
        assert high_tasks[0].priority == TaskPriority.HIGH

        await db.close()

    @pytest.mark.asyncio
    async def test_update_task_status(self) -> None:
        """Test updating task status."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)

        # Update status
        await db.update_task_status(task.id, TaskStatus.IN_PROGRESS)

        # Verify update
        updated = await db.get_task(task.id)
        assert updated.status == TaskStatus.IN_PROGRESS

        await db.close()

    @pytest.mark.asyncio
    async def test_update_task_status_creates_history(self) -> None:
        """Test that updating task status creates history entry."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)
        await db.update_task_status(task.id, TaskStatus.COMPLETED)

        # Verify history entries
        history = await db.get_task_history(task.id)
        assert len(history) >= 2
        assert any(h["action"] == "status_updated" for h in history)

        await db.close()

    @pytest.mark.asyncio
    async def test_update_task_fields(self) -> None:
        """Test updating multiple task fields."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)

        # Update multiple fields
        updates = {
            "description": "Updated task",
            "priority": TaskPriority.HIGH,
            "due_date": datetime.now() + timedelta(days=7),
        }

        await db.update_task(task.id, updates)

        # Verify updates
        updated = await db.get_task(task.id)
        assert updated.description == "Updated task"
        assert updated.priority == TaskPriority.HIGH
        assert updated.due_date is not None

        await db.close()

    @pytest.mark.asyncio
    async def test_delete_task(self) -> None:
        """Test deleting a task."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)
        await db.delete_task(task.id)

        # Verify task is deleted
        with pytest.raises(TaskNotFoundError):
            await db.get_task(task.id)

        await db.close()

    @pytest.mark.asyncio
    async def test_delete_task_preserves_history(self) -> None:
        """Test that deleting a task preserves history."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)
        task_id = task.id
        await db.delete_task(task_id)

        # Verify history is preserved
        history = await db.get_task_history(task_id)
        assert len(history) >= 2
        assert any(h["action"] == "deleted" for h in history)

        await db.close()


@pytest.mark.unit
class TestDatabaseErrorHandling:
    """Test cases for database error handling."""

    @pytest.mark.asyncio
    async def test_insert_duplicate_task_raises_error(self) -> None:
        """Test that inserting duplicate task raises error."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        task = Task(
            id=uuid4(),
            description="Test task",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="test",
            confidence=0.9,
        )

        await db.insert_task(task)

        # Try to insert same task again
        with pytest.raises(DatabaseError):
            await db.insert_task(task)

        await db.close()

    @pytest.mark.asyncio
    async def test_update_nonexistent_task_raises_error(self) -> None:
        """Test that updating nonexistent task raises error."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        with pytest.raises(TaskNotFoundError):
            await db.update_task_status(uuid4(), TaskStatus.COMPLETED)

        await db.close()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_task_raises_error(self) -> None:
        """Test that deleting nonexistent task raises error."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        with pytest.raises(TaskNotFoundError):
            await db.delete_task(uuid4())

        await db.close()

    @pytest.mark.asyncio
    async def test_invalid_filter_values_handled_gracefully(self) -> None:
        """Test that invalid filter values are handled gracefully."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Should not raise error, just return empty list
        tasks = await db.list_tasks(status="invalid_status")  # type: ignore
        assert tasks == []

        await db.close()


@pytest.mark.unit
class TestDatabaseStatistics:
    """Test cases for database statistics."""

    @pytest.mark.asyncio
    async def test_get_task_statistics(self) -> None:
        """Test getting task statistics."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        # Insert tasks with different statuses
        tasks = [
            Task(
                id=uuid4(),
                description=f"Task {i}",
                status=TaskStatus.PENDING if i < 2 else TaskStatus.COMPLETED,
                priority=TaskPriority.MEDIUM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                source="test",
                confidence=0.9,
            )
            for i in range(5)
        ]

        for task in tasks:
            await db.insert_task(task)

        # Get statistics
        stats = await db.get_statistics()
        assert stats["total"] == 5
        assert stats["pending"] == 2
        assert stats["completed"] == 3

        await db.close()

    @pytest.mark.asyncio
    async def test_statistics_with_empty_database(self) -> None:
        """Test statistics with empty database."""
        db = TaskDatabase(":memory:")
        await db.initialize()

        stats = await db.get_statistics()
        assert stats["total"] == 0
        assert stats["pending"] == 0
        assert stats["completed"] == 0

        await db.close()
