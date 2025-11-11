"""MCP Server for task management using FastMCP."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from fastmcp import FastMCP

from .config import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_MCP_HOST,
    DEFAULT_MCP_PORT,
    DEFAULT_MCP_SERVER_NAME,
)
from .database import TaskDatabase
from .exceptions import TaskNotFoundError
from .models import TaskPriority, TaskStatus
from .task_list_manager import TaskListManager

logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP(DEFAULT_MCP_SERVER_NAME)

# Global task manager (initialized in main())
_task_manager: TaskListManager | None = None


def get_task_manager() -> TaskListManager:
    """Get the global task manager instance."""
    if _task_manager is None:
        raise RuntimeError("Task manager not initialized")
    return _task_manager


def set_task_manager(task_manager: TaskListManager) -> None:
    """Set the global task manager instance (for testing)."""
    global _task_manager
    _task_manager = task_manager


async def _list_tasks_impl(
    status: str | None = None, priority: str | None = None
) -> dict[str, Any]:
    """Implementation of list_tasks tool."""
    try:
        task_manager = get_task_manager()

        # Parse filters
        status_filter = None
        priority_filter = None

        if status:
            try:
                status_filter = TaskStatus(status)
            except ValueError:
                return {"success": False, "error": f"Invalid status: {status}"}

        if priority:
            try:
                priority_filter = TaskPriority(priority)
            except ValueError:
                return {"success": False, "error": f"Invalid priority: {priority}"}

        # Get tasks
        tasks = await task_manager.list_tasks(
            status=status_filter, priority=priority_filter
        )

        # Format response
        return {
            "tasks": [
                {
                    "id": str(task.id),
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "completed_at": (
                        task.completed_at.isoformat() if task.completed_at else None
                    ),
                    "source": task.source,
                    "confidence": task.confidence,
                }
                for task in tasks
            ]
        }

    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return {"success": False, "error": str(e)}


async def _add_task_impl(
    description: str, priority: str = "medium", due_date: str | None = None
) -> dict[str, Any]:
    """
    Add a new task.

    Args:
        description: Task description (required)
        priority: Task priority (low, medium, high)
        due_date: Due date in ISO format (optional)

    Returns:
        Dictionary with task_id and success status
    """
    try:
        task_manager = get_task_manager()

        # Parse priority
        try:
            task_priority = TaskPriority(priority)
        except ValueError:
            return {"success": False, "error": f"Invalid priority: {priority}"}

        # Parse due date
        parsed_due_date = None
        if due_date:
            try:
                parsed_due_date = datetime.fromisoformat(due_date)
            except ValueError:
                return {"success": False, "error": f"Invalid date format: {due_date}"}

        # Add task
        task_id = await task_manager.add_task(
            description=description,
            priority=task_priority,
            source="mcp",
            confidence=1.0,  # MCP tasks are explicitly created
            due_date=parsed_due_date,
        )

        return {"success": True, "task_id": str(task_id)}

    except Exception as e:
        logger.error(f"Error adding task: {e}")
        return {"success": False, "error": str(e)}


async def _update_task_status_impl(task_id: str, status: str) -> dict[str, Any]:
    """
    Update the status of a task.

    Args:
        task_id: Task UUID
        status: New status (pending, in_progress, completed, cancelled)

    Returns:
        Dictionary with success status
    """
    try:
        task_manager = get_task_manager()

        # Parse task_id
        try:
            parsed_task_id = uuid.UUID(task_id)
        except ValueError:
            return {"success": False, "error": f"Invalid UUID format: {task_id}"}

        # Parse status
        try:
            task_status = TaskStatus(status)
        except ValueError:
            return {"success": False, "error": f"Invalid status: {status}"}

        # Update task
        await task_manager.update_task_status(parsed_task_id, task_status)

        return {"success": True}

    except TaskNotFoundError as e:
        logger.warning(f"Task not found: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error updating task status: {e}")
        return {"success": False, "error": str(e)}


async def _delete_task_impl(task_id: str) -> dict[str, Any]:
    """
    Delete a task.

    Args:
        task_id: Task UUID

    Returns:
        Dictionary with success status
    """
    try:
        task_manager = get_task_manager()

        # Parse task_id
        try:
            parsed_task_id = uuid.UUID(task_id)
        except ValueError:
            return {"success": False, "error": f"Invalid UUID format: {task_id}"}

        # Delete task
        await task_manager.delete_task(parsed_task_id)

        return {"success": True}

    except TaskNotFoundError as e:
        logger.warning(f"Task not found: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        return {"success": False, "error": str(e)}


async def _get_task_statistics_impl() -> dict[str, int]:
    """
    Get task statistics (counts by status).

    Returns:
        Dictionary with task counts
    """
    try:
        task_manager = get_task_manager()
        return await task_manager.get_statistics()

    except Exception as e:
        logger.error(f"Error getting task statistics: {e}")
        return {"success": False, "error": str(e)}  # type: ignore


# FastMCP decorated wrappers (for actual MCP server)
@mcp.tool()
async def list_tasks(
    status: str | None = None, priority: str | None = None
) -> dict[str, Any]:
    """
    List all tasks with optional filters.

    Args:
        status: Filter by status (pending, in_progress, completed, cancelled)
        priority: Filter by priority (low, medium, high)

    Returns:
        Dictionary with tasks list
    """
    return await _list_tasks_impl(status=status, priority=priority)


@mcp.tool()
async def add_task(
    description: str, priority: str = "medium", due_date: str | None = None
) -> dict[str, Any]:
    """
    Add a new task.

    Args:
        description: Task description (required)
        priority: Task priority (low, medium, high)
        due_date: Due date in ISO format (optional)

    Returns:
        Dictionary with task_id and success status
    """
    return await _add_task_impl(
        description=description, priority=priority, due_date=due_date
    )


@mcp.tool()
async def update_task_status(task_id: str, status: str) -> dict[str, Any]:
    """
    Update the status of a task.

    Args:
        task_id: Task UUID
        status: New status (pending, in_progress, completed, cancelled)

    Returns:
        Dictionary with success status
    """
    return await _update_task_status_impl(task_id=task_id, status=status)


@mcp.tool()
async def delete_task(task_id: str) -> dict[str, Any]:
    """
    Delete a task.

    Args:
        task_id: Task UUID

    Returns:
        Dictionary with success status
    """
    return await _delete_task_impl(task_id=task_id)


@mcp.tool()
async def get_task_statistics() -> dict[str, int]:
    """
    Get task statistics (counts by status).

    Returns:
        Dictionary with task counts
    """
    return await _get_task_statistics_impl()


# Compatibility wrapper for tests
class MCPServer:
    """
    Compatibility wrapper for testing.

    The actual MCP server uses FastMCP with function decorators.
    This class provides a compatible interface for existing tests.
    """

    def __init__(
        self,
        task_manager: TaskListManager,
        server_name: str = DEFAULT_MCP_SERVER_NAME,
        host: str = "localhost",
        port: int = 3000,
    ) -> None:
        """Initialize MCP Server wrapper."""
        self._task_manager = task_manager
        self._server_name = server_name
        self._host = host
        self._port = port
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the server."""
        set_task_manager(self._task_manager)
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the server."""
        self._initialized = False

    def get_available_tools(self) -> list[str]:
        """Get list of available tools."""
        return [
            "list_tasks",
            "add_task",
            "update_task_status",
            "delete_task",
            "get_task_statistics",
        ]

    async def handle_list_tasks(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle list_tasks request."""
        return await _list_tasks_impl(
            status=params.get("status"), priority=params.get("priority")
        )

    async def handle_add_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle add_task request."""
        if "description" not in params:
            return {"success": False, "error": "Missing required field: description"}
        return await _add_task_impl(
            description=params["description"],
            priority=params.get("priority", "medium"),
            due_date=params.get("due_date"),
        )

    async def handle_update_task_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle update_task_status request."""
        if "task_id" not in params:
            return {"success": False, "error": "Missing required field: task_id"}
        if "status" not in params:
            return {"success": False, "error": "Missing required field: status"}
        return await _update_task_status_impl(
            task_id=params["task_id"], status=params["status"]
        )

    async def handle_delete_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle delete_task request."""
        if "task_id" not in params:
            return {"success": False, "error": "Missing required field: task_id"}
        return await _delete_task_impl(task_id=params["task_id"])

    async def handle_get_task_statistics(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_task_statistics request."""
        return await _get_task_statistics_impl()


async def main(transport: str = "stdio") -> None:
    """
    Main entry point for MCP server.

    Args:
        transport: Transport type - "stdio" for stdio, "sse" for HTTP/SSE
    """
    # Initialize database
    database = TaskDatabase(DEFAULT_DATABASE_PATH)
    await database.initialize()

    # Initialize task manager
    task_manager = TaskListManager(database)
    await task_manager.initialize()
    set_task_manager(task_manager)

    logger.info(f"MCP Server initialized with 5 tools (transport={transport})")
    if transport == "sse":
        logger.info(f"Server will listen on http://{DEFAULT_MCP_HOST}:{DEFAULT_MCP_PORT}")

    try:
        # Run MCP server with specified transport
        if transport == "stdio":
            await mcp.run(transport="stdio")  # type: ignore
        else:
            # Run with HTTP/SSE transport, passing host and port
            await mcp.run(transport="sse", host=DEFAULT_MCP_HOST, port=DEFAULT_MCP_PORT)  # type: ignore
    finally:
        await task_manager.shutdown()


def cli_entry() -> None:
    """CLI entry point for the MCP server."""
    import sys

    # Check for transport argument
    transport_type = "stdio"
    if len(sys.argv) > 1 and sys.argv[1] in ("stdio", "sse", "http"):
        transport_type = "sse" if sys.argv[1] == "http" else sys.argv[1]

    # Initialize database and task manager before FastMCP takes over
    async def setup() -> None:
        database = TaskDatabase(DEFAULT_DATABASE_PATH)
        await database.initialize()
        task_manager = TaskListManager(database)
        await task_manager.initialize()
        set_task_manager(task_manager)
        logger.info(f"MCP Server initialized with 5 tools (transport={transport_type})")
        if transport_type == "sse":
            logger.info(
                f"Server will listen on http://{DEFAULT_MCP_HOST}:{DEFAULT_MCP_PORT}"
            )

    asyncio.run(setup())

    # FastMCP's run() manages its own event loop
    if transport_type == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=DEFAULT_MCP_HOST, port=DEFAULT_MCP_PORT)


if __name__ == "__main__":
    cli_entry()
