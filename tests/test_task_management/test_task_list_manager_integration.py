"""Integration tests for Task List Manager with real database."""

from datetime import datetime, timedelta

import pytest

from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.models import TaskPriority, TaskStatus
from local_ai.task_management.task_list_manager import TaskListManager


@pytest.mark.integration
@pytest.mark.asyncio
class TestTaskListManagerIntegration:
    """Integration tests with real in-memory database."""

    async def test_end_to_end_task_lifecycle(self):
        """Test complete task lifecycle from creation to deletion."""
        # Setup
        db = TaskDatabase(":memory:")
        manager = TaskListManager(db)
        await manager.initialize()

        try:
            # Create task
            task_id = await manager.add_task(
                description="Complete integration test",
                priority=TaskPriority.HIGH,
                source="test",
                confidence=0.95,
                due_date=datetime.now() + timedelta(days=7),
            )

            # Verify task exists
            task = await manager.get_task(task_id)
            assert task.description == "Complete integration test"
            assert task.status == TaskStatus.PENDING
            assert task.priority == TaskPriority.HIGH

            # Update status
            await manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)
            task = await manager.get_task(task_id)
            assert task.status == TaskStatus.IN_PROGRESS

            # Complete task
            await manager.update_task_status(task_id, TaskStatus.COMPLETED)
            task = await manager.get_task(task_id)
            assert task.status == TaskStatus.COMPLETED
            assert task.completed_at is not None

            # Check history
            history = await manager.get_task_history(task_id)
            assert len(history) >= 3  # created, status_updated x2

            # Delete task
            await manager.delete_task(task_id)

            # Verify deletion
            from local_ai.task_management.exceptions import TaskNotFoundError

            with pytest.raises(TaskNotFoundError):
                await manager.get_task(task_id)

        finally:
            await manager.shutdown()

    async def test_multiple_tasks_with_filters(self):
        """Test managing multiple tasks with filtering."""
        # Setup
        db = TaskDatabase(":memory:")
        manager = TaskListManager(db)
        await manager.initialize()

        try:
            # Create multiple tasks
            task_ids = []
            for i in range(5):
                task_id = await manager.add_task(
                    description=f"Task {i}",
                    priority=TaskPriority.HIGH if i % 2 == 0 else TaskPriority.LOW,
                    source="test",
                    confidence=0.9,
                )
                task_ids.append(task_id)

            # Update some statuses
            await manager.update_task_status(task_ids[0], TaskStatus.COMPLETED)
            await manager.update_task_status(task_ids[1], TaskStatus.IN_PROGRESS)

            # Test filtering by status
            pending_tasks = await manager.list_tasks(status=TaskStatus.PENDING)
            assert len(pending_tasks) == 3

            completed_tasks = await manager.list_tasks(status=TaskStatus.COMPLETED)
            assert len(completed_tasks) == 1

            # Test filtering by priority
            high_priority = await manager.list_tasks(priority=TaskPriority.HIGH)
            assert len(high_priority) == 3

            # Test statistics
            stats = await manager.get_statistics()
            assert stats["total"] == 5
            assert stats["pending"] == 3
            assert stats["in_progress"] == 1
            assert stats["completed"] == 1

        finally:
            await manager.shutdown()

    async def test_persistence_across_sessions(self):
        """Test that tasks persist across manager sessions."""
        import tempfile

        # Create temporary database file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            # Session 1: Create tasks
            db1 = TaskDatabase(db_path)
            manager1 = TaskListManager(db1)
            await manager1.initialize()

            task_id = await manager1.add_task(
                description="Persistent task",
                priority=TaskPriority.MEDIUM,
                source="test",
                confidence=0.85,
            )

            await manager1.shutdown()

            # Session 2: Load and verify
            db2 = TaskDatabase(db_path)
            manager2 = TaskListManager(db2)
            await manager2.initialize()

            task = await manager2.get_task(task_id)
            assert task.description == "Persistent task"
            assert task.priority == TaskPriority.MEDIUM

            stats = await manager2.get_statistics()
            assert stats["total"] == 1

            await manager2.shutdown()

        finally:
            # Cleanup
            import os

            if os.path.exists(db_path):
                os.unlink(db_path)
