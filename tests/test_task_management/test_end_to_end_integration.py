"""End-to-end integration tests for task management system."""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import pytest

from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.llm_classifier import LLMClassifier
from local_ai.task_management.mcp_server import MCPServer
from local_ai.task_management.models import ClassificationResult, TaskPriority, TaskStatus
from local_ai.task_management.task_detection_service import TaskDetectionService
from local_ai.task_management.task_list_manager import TaskListManager


@pytest.fixture
async def database() -> AsyncGenerator[TaskDatabase]:
    """Create in-memory database for testing."""
    db = TaskDatabase(":memory:", wal_mode=False)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def task_manager(
    database: TaskDatabase,
) -> AsyncGenerator[TaskListManager]:
    """Create task manager with test database."""
    manager = TaskListManager(database)
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
def ollama_available() -> bool:
    """Check if Ollama is available for testing."""
    try:
        import ollama

        client = ollama.Client()
        # Try to list models to verify connection
        models_response = client.list()
        # Check if the default model is available
        has_default_model = any(
            "llama3.2:3b" in getattr(model, "model", "")
            for model in models_response.get("models", [])
        )
        return has_default_model
    except Exception:
        return False


@pytest.fixture
async def llm_classifier(ollama_available: bool) -> Any:
    """Create LLM classifier (real or mock based on availability)."""
    if ollama_available:
        # Use real Ollama
        return LLMClassifier(timeout=15.0)
    # Return mock classifier that doesn't use Ollama

    # Create a mock that bypasses Ollama entirely
    class MockLLMClassifier:
        """Mock LLM classifier for testing without Ollama."""

        def __init__(self) -> None:
            self.model = "mock"
            self.base_url = "mock"
            self.timeout = 10.0
            self.max_retries = 3
            self.temperature = 0.1

        async def classify_text(self, text: str) -> ClassificationResult:
            """Mock classification based on simple heuristics."""
            # Simple heuristic for testing
            task_keywords = [
                "buy",
                "call",
                "email",
                "schedule",
                "remind",
                "todo",
                "task",
                "review",
                "fix",
                "write",
            ]
            is_task = any(keyword in text.lower() for keyword in task_keywords)

            return ClassificationResult(
                is_task=is_task,
                confidence=0.85 if is_task else 0.15,
                description=text if is_task else None,
                priority=TaskPriority.MEDIUM if is_task else None,
                due_date=None,
                metadata={"mock": True},
            )

    return MockLLMClassifier()


@pytest.fixture
async def task_detection_service(
    llm_classifier: Any, task_manager: TaskListManager
) -> TaskDetectionService:
    """Create task detection service."""
    return TaskDetectionService(
        llm_classifier=llm_classifier,
        task_manager=task_manager,
        confidence_threshold=0.7,
    )


@pytest.fixture
async def mcp_server(
    task_manager: TaskListManager,
) -> AsyncGenerator[MCPServer]:
    """Create MCP server."""
    server = MCPServer(task_manager=task_manager)
    await server.initialize()
    yield server
    await server.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_task_end_to_end(
    task_detection_service: TaskDetectionService, mcp_server: MCPServer
) -> None:
    """Test: Clear task text → detection → storage → MCP retrieval."""
    # Step 1: Detect task from clear text (contains "buy" keyword)
    text = "Buy groceries tomorrow"
    result = await task_detection_service.detect_task_from_text(text)

    # Verify task was detected
    assert result.task_detected is True, f"Task not detected: {result.error}"
    assert result.task is not None
    assert result.confidence >= 0.7
    assert result.error is None

    task_id = result.task.id

    # Step 2: Verify task is in storage
    task = await task_detection_service._task_manager.get_task(task_id)
    assert task.description == text or "groceries" in task.description.lower()
    assert task.status == TaskStatus.PENDING
    assert task.source == "text"

    # Step 3: Verify task is accessible via MCP
    mcp_result = await mcp_server.handle_list_tasks({})
    assert "tasks" in mcp_result
    task_ids = [uuid.UUID(t["id"]) for t in mcp_result["tasks"]]
    assert task_id in task_ids

    # Step 4: Update task via MCP
    update_result = await mcp_server.handle_update_task_status(
        {"task_id": str(task_id), "status": "completed"}
    )
    assert update_result["success"] is True

    # Step 5: Verify update
    updated_task = await task_detection_service._task_manager.get_task(task_id)
    assert updated_task.status == TaskStatus.COMPLETED
    assert updated_task.completed_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_task_text_end_to_end(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Non-task text should not create a task."""
    # Step 1: Process non-task text
    text = "The weather is nice today"
    result = await task_detection_service.detect_task_from_text(text)

    # Verify no task was created
    assert result.task_detected is False
    assert result.task is None
    assert result.error is None

    # Step 2: Verify no task in storage
    all_tasks = await task_detection_service._task_manager.list_tasks()
    assert len(all_tasks) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ambiguous_text_end_to_end(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Ambiguous text with low confidence should not create task."""
    # Step 1: Process ambiguous text
    text = "Maybe I should think about that"
    result = await task_detection_service.detect_task_from_text(text)

    # Verify behavior based on confidence
    # Either not detected as task, or detected but below threshold
    if result.task_detected:
        # If detected, confidence should be below threshold
        assert result.confidence < 0.7
        assert result.task is None
    else:
        # Not detected as task
        assert result.task is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_tasks_end_to_end(
    task_detection_service: TaskDetectionService, mcp_server: MCPServer
) -> None:
    """Test: Multiple tasks can be created and managed."""
    # Create multiple tasks
    tasks_text = [
        "Call dentist for appointment",
        "Email report to manager",
        "Schedule team meeting",
    ]

    created_task_ids = []
    for text in tasks_text:
        result = await task_detection_service.detect_task_from_text(text)
        if result.task_detected and result.task is not None:
            created_task_ids.append(result.task.id)

    # Verify at least some tasks were created
    assert len(created_task_ids) > 0

    # List all tasks via MCP
    mcp_result = await mcp_server.handle_list_tasks({})
    assert len(mcp_result["tasks"]) == len(created_task_ids)

    # Get statistics
    stats_result = await mcp_server.handle_get_task_statistics({})
    assert stats_result["total"] == len(created_task_ids)
    assert stats_result["pending"] == len(created_task_ids)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_task_lifecycle_end_to_end(
    task_detection_service: TaskDetectionService, mcp_server: MCPServer
) -> None:
    """Test: Complete task lifecycle from creation to deletion."""
    # Step 1: Create task
    text = "Review pull request #123"
    result = await task_detection_service.detect_task_from_text(text)
    assert result.task_detected is True
    assert result.task is not None
    task_id = result.task.id

    # Step 2: Update to in_progress
    await mcp_server.handle_update_task_status(
        {"task_id": str(task_id), "status": "in_progress"}
    )
    task = await task_detection_service._task_manager.get_task(task_id)
    assert task.status == TaskStatus.IN_PROGRESS

    # Step 3: Update to completed
    await mcp_server.handle_update_task_status(
        {"task_id": str(task_id), "status": "completed"}
    )
    task = await task_detection_service._task_manager.get_task(task_id)
    assert task.status == TaskStatus.COMPLETED

    # Step 4: Delete task
    delete_result = await mcp_server.handle_delete_task({"task_id": str(task_id)})
    assert delete_result["success"] is True

    # Step 5: Verify deletion
    from local_ai.task_management.exceptions import TaskNotFoundError

    with pytest.raises(TaskNotFoundError):
        await task_detection_service._task_manager.get_task(task_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_add_task_directly(mcp_server: MCPServer) -> None:
    """Test: Tasks can be added directly via MCP."""
    # Add task via MCP
    result = await mcp_server.handle_add_task(
        {
            "description": "Test task from MCP",
            "priority": "high",
            "due_date": datetime.now().isoformat(),
        }
    )

    assert result["success"] is True
    assert "task_id" in result

    # Verify task exists
    task_id = uuid.UUID(result["task_id"])
    task = await mcp_server._task_manager.get_task(task_id)
    assert task.description == "Test task from MCP"
    assert task.priority == TaskPriority.HIGH
    assert task.source == "mcp"
    assert task.confidence == 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_filter_tasks(
    task_detection_service: TaskDetectionService, mcp_server: MCPServer
) -> None:
    """Test: MCP can filter tasks by status and priority."""
    # Create tasks with different priorities
    await mcp_server.handle_add_task(
        {"description": "High priority task", "priority": "high"}
    )
    await mcp_server.handle_add_task(
        {"description": "Low priority task", "priority": "low"}
    )

    # Filter by priority
    high_priority_result = await mcp_server.handle_list_tasks({"priority": "high"})
    assert len(high_priority_result["tasks"]) == 1
    assert high_priority_result["tasks"][0]["priority"] == "high"

    low_priority_result = await mcp_server.handle_list_tasks({"priority": "low"})
    assert len(low_priority_result["tasks"]) == 1
    assert low_priority_result["tasks"][0]["priority"] == "low"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling_end_to_end(
    task_detection_service: TaskDetectionService, mcp_server: MCPServer
) -> None:
    """Test: Error handling throughout the pipeline."""
    # Test empty text
    result = await task_detection_service.detect_task_from_text("")
    assert result.task_detected is False
    assert result.error is not None

    # Test invalid task ID in MCP
    invalid_result = await mcp_server.handle_update_task_status(
        {"task_id": str(uuid.uuid4()), "status": "completed"}
    )
    assert invalid_result["success"] is False
    assert "error" in invalid_result

    # Test invalid status
    invalid_status_result = await mcp_server.handle_list_tasks({"status": "invalid"})
    assert invalid_status_result["success"] is False
    assert "error" in invalid_status_result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_ollama_integration(
    task_manager: TaskListManager, ollama_available: bool
) -> None:
    """Test: Integration with real Ollama service (if available)."""
    if not ollama_available:
        pytest.skip("Ollama not available")

    # Use real Ollama classifier
    classifier = LLMClassifier(timeout=15.0)
    service = TaskDetectionService(
        llm_classifier=classifier,
        task_manager=task_manager,
        confidence_threshold=0.7,
    )

    # Test with real LLM
    text = "Remind me to call the doctor tomorrow at 2pm"
    result = await service.detect_task_from_text(text)

    # Verify result structure (actual detection depends on LLM)
    assert result.processing_time > 0
    assert result.confidence >= 0.0
    assert result.error is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_task_operations(
    task_detection_service: TaskDetectionService, mcp_server: MCPServer
) -> None:
    """Test: Concurrent task operations work correctly."""
    # Create multiple tasks concurrently
    texts = [
        "Task 1: Buy milk",
        "Task 2: Call mom",
        "Task 3: Fix bug",
        "Task 4: Write docs",
    ]

    # Detect tasks concurrently
    results = await asyncio.gather(
        *[task_detection_service.detect_task_from_text(text) for text in texts]
    )

    # Count successful detections
    detected_count = sum(1 for r in results if r.task_detected)
    assert detected_count > 0

    # List all tasks
    mcp_result = await mcp_server.handle_list_tasks({})
    assert len(mcp_result["tasks"]) == detected_count

    # Update tasks concurrently
    task_ids = [r.task.id for r in results if r.task_detected and r.task is not None]
    await asyncio.gather(
        *[
            mcp_server.handle_update_task_status(
                {"task_id": str(tid), "status": "completed"}
            )
            for tid in task_ids
        ]
    )

    # Verify all updated
    for task_id in task_ids:
        task = await task_detection_service._task_manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
