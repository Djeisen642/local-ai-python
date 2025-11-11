"""Example demonstrating MCP Server usage."""

import asyncio
import logging

from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.mcp_server import MCPServer
from local_ai.task_management.task_list_manager import TaskListManager

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    """Demonstrate MCP Server functionality."""
    # Initialize database and task manager
    database = TaskDatabase(":memory:")
    await database.initialize()

    task_manager = TaskListManager(database)
    await task_manager.initialize()

    # Initialize MCP server
    mcp_server = MCPServer(task_manager=task_manager)
    await mcp_server.initialize()

    print("Available MCP tools:", mcp_server.get_available_tools())
    print()

    # Example 1: Add a task
    print("=== Adding a task ===")
    result = await mcp_server.handle_add_task(
        {"description": "Write documentation", "priority": "high"}
    )
    print(f"Add task result: {result}")
    task_id = result["task_id"]
    print()

    # Example 2: List tasks
    print("=== Listing all tasks ===")
    result = await mcp_server.handle_list_tasks({})
    print(f"Tasks: {result['tasks']}")
    print()

    # Example 3: Update task status
    print("=== Updating task status ===")
    result = await mcp_server.handle_update_task_status(
        {"task_id": task_id, "status": "in_progress"}
    )
    print(f"Update result: {result}")
    print()

    # Example 4: Get statistics
    print("=== Getting task statistics ===")
    stats = await mcp_server.handle_get_task_statistics({})
    print(f"Statistics: {stats}")
    print()

    # Example 5: List tasks with filter
    print("=== Listing in-progress tasks ===")
    result = await mcp_server.handle_list_tasks({"status": "in_progress"})
    print(f"In-progress tasks: {result['tasks']}")
    print()

    # Example 6: Delete task
    print("=== Deleting task ===")
    result = await mcp_server.handle_delete_task({"task_id": task_id})
    print(f"Delete result: {result}")
    print()

    # Example 7: Verify deletion
    print("=== Verifying deletion ===")
    result = await mcp_server.handle_list_tasks({})
    print(f"Remaining tasks: {result['tasks']}")
    print()

    # Cleanup
    await mcp_server.shutdown()
    await task_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
