"""Database layer for task management using SQLite."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

import aiosqlite

from local_ai.task_management.config import SCHEMA_VERSION
from local_ai.task_management.exceptions import DatabaseError, TaskNotFoundError
from local_ai.task_management.models import Task, TaskPriority, TaskStatus


class TaskDatabase:
    """SQLite database for task storage and management."""

    def __init__(self, db_path: str, wal_mode: bool = True) -> None:
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (use ":memory:" for in-memory)
            wal_mode: Enable WAL mode for concurrent access
        """
        self.db_path = db_path
        self.wal_mode = wal_mode
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database schema and connection."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row

            # Enable WAL mode for concurrent access (not supported in :memory:)
            if self.wal_mode and self.db_path != ":memory:":
                await self._connection.execute("PRAGMA journal_mode=WAL")

            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys=ON")

        # Create schema if not exists
        await self._create_schema()

    async def _create_schema(self) -> None:
        """Create database schema with tables and indexes."""
        async with self._get_connection() as conn:
            # Create schema_version table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Check current schema version
            cursor = await conn.execute("SELECT version FROM schema_version")
            result = await cursor.fetchone()
            current_version = result[0] if result else 0

            # Apply migrations if needed
            if current_version < SCHEMA_VERSION:
                await self._apply_migrations(conn, current_version)

            await conn.commit()

    async def _apply_migrations(
        self, conn: aiosqlite.Connection, from_version: int
    ) -> None:
        """
        Apply database migrations.

        Args:
            conn: Database connection
            from_version: Current schema version
        """
        if from_version < 1:
            # Create tasks table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    due_date TIMESTAMP,
                    completed_at TIMESTAMP,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT
                )
                """
            )

            # Create indexes on tasks table
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date)"
            )

            # Create task_history table
            # Note: No foreign key constraint to allow history preservation after deletion
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    action TEXT NOT NULL,
                    field_name TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    source TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )

            # Create indexes on task_history table
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_task_id ON task_history(task_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON task_history(timestamp)"
            )

            # Update schema version
            await conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Get database connection context manager.

        Yields:
            Database connection

        Raises:
            DatabaseError: If connection is not initialized
        """
        if self._connection is None:
            raise DatabaseError("Database not initialized")
        yield self._connection

    async def get_schema_version(self) -> int:
        """
        Get current schema version.

        Returns:
            Schema version number
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT version FROM schema_version")
            result = await cursor.fetchone()
            return result[0] if result else 0

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def insert_task(self, task: Task) -> None:
        """
        Insert a new task into the database.

        Args:
            task: Task to insert

        Raises:
            DatabaseError: If task already exists or insertion fails
        """
        try:
            async with self._get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO tasks (
                        id, description, status, priority, created_at, updated_at,
                        due_date, completed_at, source, confidence, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(task.id),
                        task.description,
                        task.status.value,
                        task.priority.value,
                        task.created_at.isoformat(),
                        task.updated_at.isoformat(),
                        task.due_date.isoformat() if task.due_date else None,
                        task.completed_at.isoformat() if task.completed_at else None,
                        task.source,
                        task.confidence,
                        json.dumps(task.metadata) if task.metadata else None,
                    ),
                )

                # Create history entry
                await conn.execute(
                    """
                    INSERT INTO task_history (
                        task_id, timestamp, action, source, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(task.id),
                        datetime.now().isoformat(),
                        "created",
                        task.source,
                        None,
                    ),
                )

                await conn.commit()
        except aiosqlite.IntegrityError as e:
            raise DatabaseError(f"Task with ID {task.id} already exists") from e
        except Exception as e:
            raise DatabaseError(f"Failed to insert task: {e}") from e

    async def get_task(self, task_id: UUID) -> Task:
        """
        Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task object

        Raises:
            TaskNotFoundError: If task not found
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (str(task_id),)
            )
            row = await cursor.fetchone()

            if row is None:
                raise TaskNotFoundError(f"Task with ID {task_id} not found")

            return self._row_to_task(row)

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
            List of tasks
        """
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []

        if status is not None:
            # Handle both enum and string values
            status_value = status.value if isinstance(status, TaskStatus) else status
            query += " AND status = ?"
            params.append(status_value)

        if priority is not None:
            # Handle both enum and string values
            priority_value = (
                priority.value if isinstance(priority, TaskPriority) else priority
            )
            query += " AND priority = ?"
            params.append(priority_value)

        async with self._get_connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [self._row_to_task(row) for row in rows]

    async def update_task_status(self, task_id: UUID, status: TaskStatus) -> None:
        """
        Update task status.

        Args:
            task_id: Task ID
            status: New status

        Raises:
            TaskNotFoundError: If task not found
        """
        # Get current task to verify it exists
        current_task = await self.get_task(task_id)

        async with self._get_connection() as conn:
            # Update task
            completed_at = datetime.now() if status == TaskStatus.COMPLETED else None
            await conn.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    datetime.now().isoformat(),
                    completed_at.isoformat() if completed_at else None,
                    str(task_id),
                ),
            )

            # Create history entry
            await conn.execute(
                """
                INSERT INTO task_history (
                    task_id, timestamp, action, field_name, old_value, new_value, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(task_id),
                    datetime.now().isoformat(),
                    "status_updated",
                    "status",
                    current_task.status.value,
                    status.value,
                    "system",
                ),
            )

            await conn.commit()

    async def update_task(self, task_id: UUID, updates: dict[str, Any]) -> None:
        """
        Update multiple task fields.

        Args:
            task_id: Task ID
            updates: Dictionary of field names and values to update

        Raises:
            TaskNotFoundError: If task not found
        """
        # Get current task to verify it exists
        current_task = await self.get_task(task_id)

        # Build update query
        set_clauses = ["updated_at = ?"]
        params: list[Any] = [datetime.now().isoformat()]

        for field, value in updates.items():
            if field == "priority" and isinstance(value, TaskPriority):
                set_clauses.append(f"{field} = ?")
                params.append(value.value)
            elif field == "due_date" and isinstance(value, datetime):
                set_clauses.append(f"{field} = ?")
                params.append(value.isoformat())
            else:
                set_clauses.append(f"{field} = ?")
                params.append(value)

        params.append(str(task_id))

        async with self._get_connection() as conn:
            # Update task
            query = f"UPDATE tasks SET {', '.join(set_clauses)} WHERE id = ?"
            await conn.execute(query, params)

            # Create history entries for each field
            for field, new_value in updates.items():
                old_value = getattr(current_task, field, None)
                await conn.execute(
                    """
                    INSERT INTO task_history (
                        task_id, timestamp, action, field_name, old_value, new_value, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(task_id),
                        datetime.now().isoformat(),
                        "field_updated",
                        field,
                        str(old_value) if old_value else None,
                        str(new_value) if new_value else None,
                        "system",
                    ),
                )

            await conn.commit()

    async def delete_task(self, task_id: UUID) -> None:
        """
        Delete a task.

        Args:
            task_id: Task ID

        Raises:
            TaskNotFoundError: If task not found
        """
        # Verify task exists
        await self.get_task(task_id)

        async with self._get_connection() as conn:
            # Create history entry before deletion
            await conn.execute(
                """
                INSERT INTO task_history (
                    task_id, timestamp, action, source
                ) VALUES (?, ?, ?, ?)
                """,
                (str(task_id), datetime.now().isoformat(), "deleted", "system"),
            )

            # Delete task
            await conn.execute("DELETE FROM tasks WHERE id = ?", (str(task_id),))

            await conn.commit()

    async def get_task_history(self, task_id: UUID) -> list[dict[str, Any]]:
        """
        Get task history.

        Args:
            task_id: Task ID

        Returns:
            List of history entries
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT timestamp, action, field_name, old_value, new_value, source
                FROM task_history
                WHERE task_id = ?
                ORDER BY timestamp ASC
                """,
                (str(task_id),),
            )
            rows = await cursor.fetchall()

            return [
                {
                    "timestamp": row[0],
                    "action": row[1],
                    "field_name": row[2],
                    "old_value": row[3],
                    "new_value": row[4],
                    "source": row[5],
                }
                for row in rows
            ]

    async def get_statistics(self) -> dict[str, int]:
        """
        Get task statistics.

        Returns:
            Dictionary with task counts by status
        """
        async with self._get_connection() as conn:
            # Get total count
            cursor = await conn.execute("SELECT COUNT(*) FROM tasks")
            total = (await cursor.fetchone())[0]

            # Get counts by status
            cursor = await conn.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            )
            status_counts = {row[0]: row[1] for row in await cursor.fetchall()}

            return {
                "total": total,
                "pending": status_counts.get(TaskStatus.PENDING.value, 0),
                "in_progress": status_counts.get(TaskStatus.IN_PROGRESS.value, 0),
                "completed": status_counts.get(TaskStatus.COMPLETED.value, 0),
                "cancelled": status_counts.get(TaskStatus.CANCELLED.value, 0),
            }

    def _row_to_task(self, row: aiosqlite.Row) -> Task:
        """
        Convert database row to Task object.

        Args:
            row: Database row

        Returns:
            Task object
        """
        return Task(
            id=UUID(row["id"]),
            description=row["description"],
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            due_date=(
                datetime.fromisoformat(row["due_date"]) if row["due_date"] else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            source=row["source"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
