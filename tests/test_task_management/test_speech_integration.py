"""Integration tests for speech-to-text and task detection integration."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from local_ai.speech_to_text.models import TranscriptionResult
from local_ai.speech_to_text.service import SpeechToTextService
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.models import ClassificationResult, TaskPriority
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
def mock_llm_classifier() -> Any:
    """Create mock LLM classifier."""

    class MockLLMClassifier:
        """Mock LLM classifier for testing."""

        async def classify_text(self, text: str) -> ClassificationResult:
            """Mock classification based on keywords."""
            task_keywords = [
                "buy",
                "call",
                "email",
                "schedule",
                "remind",
                "todo",
                "task",
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
    mock_llm_classifier: Any, task_manager: TaskListManager
) -> TaskDetectionService:
    """Create task detection service."""
    return TaskDetectionService(
        llm_classifier=mock_llm_classifier,
        task_manager=task_manager,
        confidence_threshold=0.7,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_task_detection_integration(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Transcription triggers task detection."""
    # Create speech-to-text service with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        # Simulate transcription result with task
        transcription_result = TranscriptionResult(
            text="Buy groceries tomorrow",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # Trigger transcription update
        service._update_transcription_with_result(transcription_result)

        # Wait for async task detection to complete
        await asyncio.sleep(0.1)

        # Verify task was created
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 1
        assert "groceries" in tasks[0].description.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_non_task_transcription(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Non-task transcription doesn't create tasks."""
    # Create speech-to-text service with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        # Simulate transcription result without task
        transcription_result = TranscriptionResult(
            text="The weather is nice today",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # Trigger transcription update
        service._update_transcription_with_result(transcription_result)

        # Wait for async task detection to complete
        await asyncio.sleep(0.1)

        # Verify no task was created
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_task_detection_disabled(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Task detection can be disabled."""
    # Create speech-to-text service with task detection disabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=False, task_detection_service=task_detection_service
        )

        # Simulate transcription result with task
        transcription_result = TranscriptionResult(
            text="Buy groceries tomorrow",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # Trigger transcription update
        service._update_transcription_with_result(transcription_result)

        # Wait to ensure no task detection happens
        await asyncio.sleep(0.1)

        # Verify no task was created (detection disabled)
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_task_detection_non_blocking(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Task detection doesn't block transcription processing."""
    # Create speech-to-text service with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        # Track callback execution
        callback_called = False
        callback_result = None

        def transcription_callback(result: TranscriptionResult) -> None:
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result

        service.set_transcription_result_callback(transcription_callback)

        # Simulate transcription result
        transcription_result = TranscriptionResult(
            text="Call dentist tomorrow",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # Trigger transcription update
        service._update_transcription_with_result(transcription_result)

        # Callback should be called immediately (non-blocking)
        assert callback_called is True
        assert callback_result is not None
        assert callback_result.text == "Call dentist tomorrow"

        # Wait for async task detection to complete
        await asyncio.sleep(0.1)

        # Verify task was created
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_multiple_transcriptions(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Multiple transcriptions create multiple tasks."""
    # Create speech-to-text service with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        # Simulate multiple transcription results
        transcriptions = [
            "Buy milk",
            "Call mom",
            "Email report",
        ]

        for text in transcriptions:
            transcription_result = TranscriptionResult(
                text=text,
                confidence=0.9,
                timestamp=1234567890.0,
                processing_time=0.5,
            )
            service._update_transcription_with_result(transcription_result)

        # Wait for all async task detections to complete
        await asyncio.sleep(0.2)

        # Verify all tasks were created
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_task_detection_error_handling(
    task_manager: TaskListManager,
) -> None:
    """Test: Task detection errors don't crash the service."""
    # Create a failing task detection service
    failing_service = AsyncMock(spec=TaskDetectionService)
    failing_service.detect_task_from_text.side_effect = Exception("Task detection failed")

    # Create speech-to-text service with failing task detection
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=failing_service
        )

        # Simulate transcription result
        transcription_result = TranscriptionResult(
            text="Buy groceries",
            confidence=0.9,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # This should not raise an exception
        service._update_transcription_with_result(transcription_result)

        # Wait for async task detection to complete
        await asyncio.sleep(0.1)

        # Service should still be functional
        assert service.get_latest_transcription_result() is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_empty_transcription(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Empty transcriptions don't trigger task detection."""
    # Create speech-to-text service with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        # Simulate empty transcription result
        transcription_result = TranscriptionResult(
            text="",
            confidence=0.0,
            timestamp=1234567890.0,
            processing_time=0.5,
        )

        # Trigger transcription update
        service._update_transcription_with_result(transcription_result)

        # Wait to ensure no task detection happens
        await asyncio.sleep(0.1)

        # Verify no task was created
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_configuration_toggle(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Task detection can be toggled via configuration."""
    # Test with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service_enabled = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        assert service_enabled._enable_task_detection is True
        assert service_enabled._task_detection_service is not None

        # Test with task detection disabled
        service_disabled = SpeechToTextService(
            enable_task_detection=False, task_detection_service=None
        )

        assert service_disabled._enable_task_detection is False
        assert service_disabled._task_detection_service is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_speech_to_text_task_detection_with_metadata(
    task_detection_service: TaskDetectionService,
) -> None:
    """Test: Task detection works with transcription metadata."""
    # Create speech-to-text service with task detection enabled
    with patch(
        "src.local_ai.speech_to_text.service.SpeechToTextService._initialize_components"
    ) as mock_init:
        mock_init.return_value = True

        service = SpeechToTextService(
            enable_task_detection=True, task_detection_service=task_detection_service
        )

        # Simulate transcription result with metadata
        transcription_result = TranscriptionResult(
            text="Schedule meeting with team",
            confidence=0.95,
            timestamp=1234567890.0,
            processing_time=0.3,
        )

        metadata = {
            "confidence": 0.95,
            "timestamp": 1234567890.0,
            "processing_time": 0.3,
            "audio_duration": 2.5,
            "sample_rate": 16000,
            "chunk_count": 10,
        }

        # Trigger transcription update with metadata
        service._update_transcription_with_result(transcription_result, metadata)

        # Wait for async task detection to complete
        await asyncio.sleep(0.1)

        # Verify task was created
        tasks = await task_detection_service._task_manager.list_tasks()
        assert len(tasks) == 1
        assert "meeting" in tasks[0].description.lower()
