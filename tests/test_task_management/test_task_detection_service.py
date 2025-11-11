"""Unit tests for Task Detection Service (TDD - RED phase)."""

import uuid
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from local_ai.task_management.exceptions import ClassificationError
from local_ai.task_management.models import (
    ClassificationResult,
    TaskDetectionResult,
    TaskPriority,
)


@pytest.fixture
def mock_llm_classifier() -> AsyncMock:
    """Create a mock LLM classifier for testing."""
    classifier = AsyncMock()
    classifier.classify_text = AsyncMock()
    return classifier


@pytest.fixture
def mock_task_manager() -> AsyncMock:
    """Create a mock task manager for testing."""
    manager = AsyncMock()
    manager.add_task = AsyncMock(return_value=uuid.uuid4())
    manager.get_task = AsyncMock()
    manager.list_tasks = AsyncMock(return_value=[])
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


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskDetectionServiceInitialization:
    """Test Task Detection Service initialization."""

    async def test_initialization_with_dependencies(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test service initializes with LLM classifier and task manager."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        assert service._llm_classifier == mock_llm_classifier
        assert service._task_manager == mock_task_manager

    async def test_initialization_with_custom_confidence_threshold(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test service initializes with custom confidence threshold."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier,
            task_manager=mock_task_manager,
            confidence_threshold=0.8,
        )

        assert service._confidence_threshold == 0.8

    async def test_initialization_with_default_confidence_threshold(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test service uses default confidence threshold."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        assert service._confidence_threshold == 0.7  # DEFAULT_CONFIDENCE_THRESHOLD


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskDetectionFlow:
    """Test task detection flow with mocked dependencies."""

    async def test_detect_task_from_clear_task_text(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test detecting task from clear task text."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classification result
        classification = ClassificationResult(
            is_task=True,
            confidence=0.95,
            description="Buy groceries tomorrow",
            priority=TaskPriority.HIGH,
            due_date=datetime.now() + timedelta(days=1),
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("buy groceries tomorrow")

        assert result.task_detected is True
        assert result.task is not None
        assert result.confidence == 0.95
        mock_llm_classifier.classify_text.assert_called_once_with(
            "buy groceries tomorrow"
        )
        mock_task_manager.add_task.assert_called_once()

    async def test_detect_task_from_non_task_text(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test detecting non-task from text."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classification result
        classification = ClassificationResult(
            is_task=False, confidence=0.9, description=None, priority=None
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("the weather is nice today")

        assert result.task_detected is False
        assert result.task is None
        assert result.confidence == 0.9
        mock_llm_classifier.classify_text.assert_called_once()
        mock_task_manager.add_task.assert_not_called()

    async def test_detect_task_with_metadata_extraction(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test task detection extracts metadata correctly."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classification result with metadata
        due_date = datetime.now() + timedelta(days=3)
        classification = ClassificationResult(
            is_task=True,
            confidence=0.92,
            description="Submit report by Friday",
            priority=TaskPriority.HIGH,
            due_date=due_date,
            metadata={"category": "work"},
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("submit report by Friday")

        assert result.task_detected is True
        mock_task_manager.add_task.assert_called_once()
        call_kwargs = mock_task_manager.add_task.call_args[1]
        assert call_kwargs["description"] == "Submit report by Friday"
        assert call_kwargs["priority"] == TaskPriority.HIGH
        assert call_kwargs["due_date"] == due_date

    async def test_detect_task_tracks_processing_time(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test task detection tracks processing time."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("test text")

        assert result.processing_time > 0
        assert isinstance(result.processing_time, float)


@pytest.mark.unit
@pytest.mark.asyncio
class TestConfidenceThresholding:
    """Test confidence threshold filtering."""

    async def test_task_below_threshold_not_created(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test task below confidence threshold is not created."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classification with low confidence
        classification = ClassificationResult(
            is_task=True,
            confidence=0.5,  # Below default threshold of 0.7
            description="Maybe do something",
            priority=TaskPriority.LOW,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("maybe do something")

        assert result.task_detected is False
        assert result.task is None
        assert result.confidence == 0.5
        mock_task_manager.add_task.assert_not_called()

    async def test_task_at_threshold_is_created(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test task at confidence threshold is created."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classification at threshold
        classification = ClassificationResult(
            is_task=True,
            confidence=0.7,  # At default threshold
            description="Do something",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("do something")

        assert result.task_detected is True
        assert result.task is not None
        mock_task_manager.add_task.assert_called_once()

    async def test_custom_threshold_respected(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test custom confidence threshold is respected."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classification
        classification = ClassificationResult(
            is_task=True,
            confidence=0.75,
            description="Do something",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        # Service with custom threshold of 0.8
        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier,
            task_manager=mock_task_manager,
            confidence_threshold=0.8,
        )

        result = await service.detect_task_from_text("do something")

        # 0.75 < 0.8, so task should not be created
        assert result.task_detected is False
        mock_task_manager.add_task.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in task detection."""

    async def test_classification_error_returns_error_result(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test classification error returns error result without crashing."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock LLM classifier to raise error
        mock_llm_classifier.classify_text.side_effect = ClassificationError(
            "LLM service unavailable"
        )

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("test text")

        assert result.task_detected is False
        assert result.task is None
        assert result.error is not None
        assert "LLM service unavailable" in result.error
        mock_task_manager.add_task.assert_not_called()

    async def test_empty_text_handled_gracefully(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test empty text is handled gracefully."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("")

        assert result.task_detected is False
        assert result.error is not None
        mock_llm_classifier.classify_text.assert_not_called()
        mock_task_manager.add_task.assert_not_called()

    async def test_whitespace_only_text_handled_gracefully(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test whitespace-only text is handled gracefully."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("   \n\t  ")

        assert result.task_detected is False
        assert result.error is not None
        mock_llm_classifier.classify_text.assert_not_called()

    async def test_task_manager_error_returns_error_result(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test task manager error returns error result."""
        from local_ai.task_management.exceptions import DatabaseError
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock successful classification
        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        # Mock task manager to raise error
        mock_task_manager.add_task.side_effect = DatabaseError("Database error")

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("test text")

        assert result.task_detected is False
        assert result.error is not None
        assert "Database error" in result.error

    async def test_unexpected_error_handled_gracefully(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test unexpected errors are handled gracefully."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        # Mock unexpected error
        mock_llm_classifier.classify_text.side_effect = Exception("Unexpected error")

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("test text")

        assert result.task_detected is False
        assert result.error is not None


@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncProcessing:
    """Test async processing capabilities."""

    async def test_detect_task_is_async(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test detect_task_from_text is async."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        # Should be awaitable
        result = await service.detect_task_from_text("test text")
        assert isinstance(result, TaskDetectionResult)

    async def test_concurrent_requests_supported(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test service supports concurrent requests."""
        import asyncio

        from local_ai.task_management.task_detection_service import TaskDetectionService

        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        # Run multiple detections concurrently
        results = await asyncio.gather(
            service.detect_task_from_text("task 1"),
            service.detect_task_from_text("task 2"),
            service.detect_task_from_text("task 3"),
        )

        assert len(results) == 3
        assert all(r.task_detected for r in results)
        assert mock_llm_classifier.classify_text.call_count == 3

    async def test_error_in_one_request_doesnt_affect_others(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test error in one request doesn't affect concurrent requests."""
        import asyncio

        from local_ai.task_management.task_detection_service import TaskDetectionService

        # First call fails, others succeed
        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.side_effect = [
            ClassificationError("Error"),
            classification,
            classification,
        ]

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        results = await asyncio.gather(
            service.detect_task_from_text("task 1"),
            service.detect_task_from_text("task 2"),
            service.detect_task_from_text("task 3"),
        )

        assert len(results) == 3
        assert results[0].task_detected is False  # Error
        assert results[1].task_detected is True  # Success
        assert results[2].task_detected is True  # Success


@pytest.mark.unit
@pytest.mark.asyncio
class TestLoggingAndMetrics:
    """Test logging and metrics tracking."""

    async def test_task_detection_logged(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test task detection events are logged."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        with patch("logging.Logger.info") as mock_log:
            await service.detect_task_from_text("test text")
            assert mock_log.called

    async def test_processing_time_tracked(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test processing time is tracked."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        classification = ClassificationResult(
            is_task=True,
            confidence=0.9,
            description="Test task",
            priority=TaskPriority.MEDIUM,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        result = await service.detect_task_from_text("test text")

        assert result.processing_time >= 0
        assert isinstance(result.processing_time, float)

    async def test_error_logged(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test errors are logged."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        mock_llm_classifier.classify_text.side_effect = ClassificationError("Test error")

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        with patch("logging.Logger.error") as mock_log:
            await service.detect_task_from_text("test text")
            assert mock_log.called

    async def test_confidence_score_logged(
        self, mock_llm_classifier: Any, mock_task_manager: Any
    ) -> None:
        """Test confidence score is logged."""
        from local_ai.task_management.task_detection_service import TaskDetectionService

        classification = ClassificationResult(
            is_task=True,
            confidence=0.95,
            description="Test task",
            priority=TaskPriority.HIGH,
        )
        mock_llm_classifier.classify_text.return_value = classification

        service = TaskDetectionService(
            llm_classifier=mock_llm_classifier, task_manager=mock_task_manager
        )

        with patch("logging.Logger.info") as mock_log:
            await service.detect_task_from_text("test text")
            # Check that confidence was logged
            assert mock_log.called
            log_message = str(mock_log.call_args)
            assert "0.95" in log_message or "confidence" in log_message.lower()
