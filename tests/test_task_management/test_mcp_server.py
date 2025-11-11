"""Unit tests for MCP Server."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from local_ai.task_management.exceptions import TaskNotFoundError
from local_ai.task_management.models import Task, TaskPriority, TaskStatus


@pytest.fixture
def mock_task_manager() -> AsyncMock:
    """Create a mock Task List Manager."""
    manager = AsyncMock()
    manager.list_tasks = AsyncMock(return_value=[])
    manager.add_task = AsyncMock(return_value=uuid.uuid4())
    manager.update_task_status = AsyncMock()
    manager.delete_task = AsyncMock()
    manager.get_statistics = AsyncMock(
        return_value={
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
        }
    )
    return manager


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    task_id = uuid.uuid4()
    now = datetime.now()
    return Task(
        id=task_id,
        description="Test task",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        created_at=now,
        updated_at=now,
        source="test",
        confidence=0.9,
    )


@pytest.mark.unit
class TestMCPServerInitialization:
    """Test MCP server initialization."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_task_manager: AsyncMock) -> None:
        """Test that MCP server initializes correctly."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        assert server._task_manager == mock_task_manager
        assert server._initialized is True

    @pytest.mark.asyncio
    async def test_server_initialization_with_config(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test server initialization with custom configuration."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(
            task_manager=mock_task_manager,
            server_name="test-server",
            host="0.0.0.0",
            port=4000,
        )
        await server.initialize()

        assert server._server_name == "test-server"
        assert server._host == "0.0.0.0"
        assert server._port == 4000

    @pytest.mark.asyncio
    async def test_server_shutdown(self, mock_task_manager: AsyncMock) -> None:
        """Test server shutdown."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()
        await server.shutdown()

        assert server._initialized is False


@pytest.mark.unit
class TestMCPToolRegistration:
    """Test MCP tool registration."""

    @pytest.mark.asyncio
    async def test_list_tasks_tool_registered(self, mock_task_manager: AsyncMock) -> None:
        """Test that list_tasks tool is registered."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        tools = server.get_available_tools()
        assert "list_tasks" in tools

    @pytest.mark.asyncio
    async def test_add_task_tool_registered(self, mock_task_manager: AsyncMock) -> None:
        """Test that add_task tool is registered."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        tools = server.get_available_tools()
        assert "add_task" in tools

    @pytest.mark.asyncio
    async def test_update_task_status_tool_registered(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test that update_task_status tool is registered."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        tools = server.get_available_tools()
        assert "update_task_status" in tools

    @pytest.mark.asyncio
    async def test_delete_task_tool_registered(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test that delete_task tool is registered."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        tools = server.get_available_tools()
        assert "delete_task" in tools

    @pytest.mark.asyncio
    async def test_get_task_statistics_tool_registered(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test that get_task_statistics tool is registered."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        tools = server.get_available_tools()
        assert "get_task_statistics" in tools


@pytest.mark.unit
class TestListTasksTool:
    """Test list_tasks MCP tool."""

    @pytest.mark.asyncio
    async def test_list_all_tasks(
        self, mock_task_manager: AsyncMock, sample_task: Task
    ) -> None:
        """Test listing all tasks."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.list_tasks.return_value = [sample_task]

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_list_tasks({})

        mock_task_manager.list_tasks.assert_called_once_with(status=None, priority=None)
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["id"] == str(sample_task.id)
        assert result["tasks"][0]["description"] == sample_task.description

    @pytest.mark.asyncio
    async def test_list_tasks_with_status_filter(
        self, mock_task_manager: AsyncMock, sample_task: Task
    ) -> None:
        """Test listing tasks with status filter."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.list_tasks.return_value = [sample_task]

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_list_tasks({"status": "pending"})

        mock_task_manager.list_tasks.assert_called_once_with(
            status=TaskStatus.PENDING, priority=None
        )
        assert len(result["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_list_tasks_with_priority_filter(
        self, mock_task_manager: AsyncMock, sample_task: Task
    ) -> None:
        """Test listing tasks with priority filter."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.list_tasks.return_value = [sample_task]

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_list_tasks({"priority": "high"})

        mock_task_manager.list_tasks.assert_called_once_with(
            status=None, priority=TaskPriority.HIGH
        )
        assert len(result["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, mock_task_manager: AsyncMock) -> None:
        """Test listing tasks when no tasks exist."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.list_tasks.return_value = []

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_list_tasks({})

        assert result["tasks"] == []


@pytest.mark.unit
class TestAddTaskTool:
    """Test add_task MCP tool."""

    @pytest.mark.asyncio
    async def test_add_task_minimal(self, mock_task_manager: AsyncMock) -> None:
        """Test adding a task with minimal parameters."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()
        mock_task_manager.add_task.return_value = task_id

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_add_task({"description": "New task"})

        mock_task_manager.add_task.assert_called_once()
        assert result["task_id"] == str(task_id)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_add_task_with_priority(self, mock_task_manager: AsyncMock) -> None:
        """Test adding a task with priority."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()
        mock_task_manager.add_task.return_value = task_id

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_add_task(
            {"description": "New task", "priority": "high"}
        )

        call_args = mock_task_manager.add_task.call_args
        assert call_args.kwargs["priority"] == TaskPriority.HIGH
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_add_task_with_due_date(self, mock_task_manager: AsyncMock) -> None:
        """Test adding a task with due date."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()
        mock_task_manager.add_task.return_value = task_id

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        due_date = "2024-12-31T23:59:59"
        result = await server.handle_add_task(
            {"description": "New task", "due_date": due_date}
        )

        call_args = mock_task_manager.add_task.call_args
        assert call_args.kwargs["due_date"] is not None
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_add_task_missing_description(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test adding a task without description fails validation."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_add_task({})

        assert result["success"] is False
        assert "error" in result
        mock_task_manager.add_task.assert_not_called()


@pytest.mark.unit
class TestUpdateTaskStatusTool:
    """Test update_task_status MCP tool."""

    @pytest.mark.asyncio
    async def test_update_task_status(self, mock_task_manager: AsyncMock) -> None:
        """Test updating task status."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_update_task_status(
            {"task_id": str(task_id), "status": "completed"}
        )

        mock_task_manager.update_task_status.assert_called_once_with(
            task_id, TaskStatus.COMPLETED
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_update_task_status_invalid_task(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test updating status of non-existent task."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()
        mock_task_manager.update_task_status.side_effect = TaskNotFoundError(
            f"Task {task_id} not found"
        )

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_update_task_status(
            {"task_id": str(task_id), "status": "completed"}
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_task_status_missing_params(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test updating task status with missing parameters."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_update_task_status({})

        assert result["success"] is False
        assert "error" in result
        mock_task_manager.update_task_status.assert_not_called()


@pytest.mark.unit
class TestDeleteTaskTool:
    """Test delete_task MCP tool."""

    @pytest.mark.asyncio
    async def test_delete_task(self, mock_task_manager: AsyncMock) -> None:
        """Test deleting a task."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_delete_task({"task_id": str(task_id)})

        mock_task_manager.delete_task.assert_called_once_with(task_id)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_task_not_found(self, mock_task_manager: AsyncMock) -> None:
        """Test deleting a non-existent task."""
        from local_ai.task_management.mcp_server import MCPServer

        task_id = uuid.uuid4()
        mock_task_manager.delete_task.side_effect = TaskNotFoundError(
            f"Task {task_id} not found"
        )

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_delete_task({"task_id": str(task_id)})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_delete_task_missing_id(self, mock_task_manager: AsyncMock) -> None:
        """Test deleting a task without providing ID."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_delete_task({})

        assert result["success"] is False
        assert "error" in result
        mock_task_manager.delete_task.assert_not_called()


@pytest.mark.unit
class TestGetTaskStatisticsTool:
    """Test get_task_statistics MCP tool."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, mock_task_manager: AsyncMock) -> None:
        """Test getting task statistics."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.get_statistics.return_value = {
            "total": 10,
            "pending": 3,
            "in_progress": 2,
            "completed": 4,
            "cancelled": 1,
        }

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_get_task_statistics({})

        mock_task_manager.get_statistics.assert_called_once()
        assert result["total"] == 10
        assert result["pending"] == 3
        assert result["in_progress"] == 2
        assert result["completed"] == 4
        assert result["cancelled"] == 1


@pytest.mark.unit
class TestRequestValidation:
    """Test MCP request validation."""

    @pytest.mark.asyncio
    async def test_validate_invalid_status(self, mock_task_manager: AsyncMock) -> None:
        """Test validation rejects invalid status values."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_list_tasks({"status": "invalid_status"})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_invalid_priority(self, mock_task_manager: AsyncMock) -> None:
        """Test validation rejects invalid priority values."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_add_task(
            {"description": "Test", "priority": "invalid_priority"}
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_invalid_uuid(self, mock_task_manager: AsyncMock) -> None:
        """Test validation rejects invalid UUID format."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_delete_task({"task_id": "not-a-uuid"})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_invalid_date_format(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test validation rejects invalid date format."""
        from local_ai.task_management.mcp_server import MCPServer

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_add_task(
            {"description": "Test", "due_date": "not-a-date"}
        )

        assert result["success"] is False
        assert "error" in result


@pytest.mark.unit
class TestErrorHandling:
    """Test MCP server error handling."""

    @pytest.mark.asyncio
    async def test_handle_database_error(self, mock_task_manager: AsyncMock) -> None:
        """Test handling database errors gracefully."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.list_tasks.side_effect = Exception("Database error")

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_list_tasks({})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_unexpected_error(self, mock_task_manager: AsyncMock) -> None:
        """Test handling unexpected errors."""
        from local_ai.task_management.mcp_server import MCPServer

        mock_task_manager.add_task.side_effect = RuntimeError("Unexpected error")

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        result = await server.handle_add_task({"description": "Test"})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_server_continues_after_error(
        self, mock_task_manager: AsyncMock
    ) -> None:
        """Test that server continues operating after an error."""
        from local_ai.task_management.mcp_server import MCPServer

        # First call fails
        mock_task_manager.list_tasks.side_effect = [
            Exception("Error"),
            [],  # Second call succeeds
        ]

        server = MCPServer(task_manager=mock_task_manager)
        await server.initialize()

        # First call should fail
        result1 = await server.handle_list_tasks({})
        assert result1["success"] is False

        # Second call should succeed
        result2 = await server.handle_list_tasks({})
        assert "tasks" in result2
