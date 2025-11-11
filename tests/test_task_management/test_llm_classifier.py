"""Tests for LLM classifier functionality."""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from local_ai.task_management.exceptions import ClassificationError
from local_ai.task_management.models import TaskPriority

# Test constants
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
CUSTOM_TIMEOUT = 5.0
CUSTOM_MAX_RETRIES = 5
TEST_CONFIDENCE_HIGH = 0.95
TEST_CONFIDENCE_MEDIUM = 0.9
TEST_CONFIDENCE_LOW = 0.6
CONFIDENCE_THRESHOLD = 0.7


def to_toml(data: dict[str, Any]) -> str:
    """Convert dict to TOML format for testing."""
    lines = []
    for key, value in data.items():
        if value is None or value == "null":
            continue  # Skip null values in TOML
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
    return "\n".join(lines)


@pytest.mark.unit
class TestOllamaConnection:
    """Test cases for Ollama connection management."""

    @pytest.mark.asyncio
    async def test_classifier_initialization_with_default_config(self) -> None:
        """Test classifier can be initialized with default configuration."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        assert classifier.model == "llama3.2:1b"
        assert classifier.base_url == "http://localhost:11434"
        assert classifier.timeout == DEFAULT_TIMEOUT
        assert classifier.max_retries == DEFAULT_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_classifier_initialization_with_custom_config(self) -> None:
        """Test classifier can be initialized with custom configuration."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(
            model="custom-model",
            base_url="http://custom:8080",
            timeout=CUSTOM_TIMEOUT,
            max_retries=CUSTOM_MAX_RETRIES,
        )
        assert classifier.model == "custom-model"
        assert classifier.base_url == "http://custom:8080"
        assert classifier.timeout == CUSTOM_TIMEOUT
        assert classifier.max_retries == CUSTOM_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_ollama_connection_error_raises_classification_error(self) -> None:
        """Test that Ollama connection errors raise ClassificationError."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        with patch(
            "ollama.AsyncClient.chat", side_effect=ConnectionError("Connection failed")
        ):
            with pytest.raises(ClassificationError, match="Connection failed"):
                await classifier.classify_text("test text")

    @pytest.mark.asyncio
    async def test_ollama_unavailable_raises_classification_error(self) -> None:
        """Test that unavailable Ollama service raises ClassificationError."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        with patch(
            "ollama.AsyncClient.chat",
            side_effect=Exception("Ollama service unavailable"),
        ):
            with pytest.raises(ClassificationError):
                await classifier.classify_text("test text")


@pytest.mark.unit
class TestPromptGeneration:
    """Test cases for prompt generation."""

    @pytest.mark.asyncio
    async def test_prompt_includes_task_identification_instructions(self) -> None:
        """Test that prompt includes instructions for task identification."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        prompt = classifier._generate_prompt("buy groceries tomorrow")

        assert "task" in prompt.lower()
        assert "actionable" in prompt.lower() or "action" in prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_includes_toml_format_instructions(self) -> None:
        """Test that prompt includes TOML format instructions."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        prompt = classifier._generate_prompt("test text")

        assert "toml" in prompt.lower()
        assert "is_task" in prompt.lower()
        assert "confidence" in prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_includes_input_text(self) -> None:
        """Test that prompt includes the input text."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        input_text = "buy groceries tomorrow"
        prompt = classifier._generate_prompt(input_text)

        assert input_text in prompt

    @pytest.mark.asyncio
    async def test_prompt_requests_priority_extraction(self) -> None:
        """Test that prompt requests priority extraction."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        prompt = classifier._generate_prompt("urgent: fix the bug")

        assert "priority" in prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_requests_due_date_extraction(self) -> None:
        """Test that prompt requests due date extraction."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        prompt = classifier._generate_prompt("submit report by Friday")

        assert "due" in prompt.lower() or "date" in prompt.lower()


@pytest.mark.unit
class TestResponseParsing:
    """Test cases for response parsing."""

    @pytest.mark.asyncio
    async def test_parse_valid_task_response(self) -> None:
        """Test parsing valid task response."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        response = """is_task = true
confidence = 0.95
description = "Buy groceries"
priority = "high"
"""

        result = classifier._parse_response(response)

        assert result.is_task is True
        assert result.confidence == TEST_CONFIDENCE_HIGH
        assert result.description == "Buy groceries"
        assert result.priority == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_parse_valid_non_task_response(self) -> None:
        """Test parsing valid non-task response."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        response = """is_task = false
confidence = 0.85
"""

        result = classifier._parse_response(response)

        assert result.is_task is False
        assert result.confidence == 0.85
        assert result.description is None
        assert result.priority is None

    @pytest.mark.asyncio
    async def test_parse_response_with_due_date(self) -> None:
        """Test parsing response with due date."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        due_date_str = (datetime.now() + timedelta(days=1)).isoformat()
        response = f"""is_task = true
confidence = 0.9
description = "Submit report"
priority = "medium"
due_date = "{due_date_str}"
"""

        result = classifier._parse_response(response)

        assert result.is_task is True
        assert result.due_date is not None
        assert isinstance(result.due_date, datetime)

    @pytest.mark.asyncio
    async def test_parse_invalid_toml_raises_error(self) -> None:
        """Test that invalid TOML raises ClassificationError."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        with pytest.raises(ClassificationError, match="Invalid TOML"):
            classifier._parse_response("not valid toml [[[[")

    @pytest.mark.asyncio
    async def test_parse_missing_required_fields_raises_error(self) -> None:
        """Test that missing required fields raises ClassificationError."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        response = "confidence = 0.9"  # Missing is_task

        with pytest.raises(ClassificationError, match="Missing required field"):
            classifier._parse_response(response)

    @pytest.mark.asyncio
    async def test_parse_invalid_priority_uses_medium_default(self) -> None:
        """Test that invalid priority defaults to medium."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        response = """is_task = true
confidence = 0.9
description = "Test task"
priority = "invalid_priority"
"""

        result = classifier._parse_response(response)

        assert result.priority == TaskPriority.MEDIUM

    @pytest.mark.asyncio
    async def test_parse_invalid_due_date_sets_none(self) -> None:
        """Test that invalid due date is set to None."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        response = """is_task = true
confidence = 0.9
description = "Test task"
priority = "high"
due_date = "invalid date"
"""

        result = classifier._parse_response(response)

        assert result.due_date is None

    @pytest.mark.asyncio
    async def test_parse_response_with_metadata(self) -> None:
        """Test parsing response with additional metadata."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()
        response = """is_task = true
confidence = 0.9
description = "Test task"
priority = "high"
extra_field = "extra_value"
"""

        result = classifier._parse_response(response)

        assert result.metadata.get("extra_field") == "extra_value"


@pytest.mark.unit
class TestRetryLogic:
    """Test cases for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        """Test that classifier retries on timeout."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(max_retries=DEFAULT_MAX_RETRIES)

        call_count = 0
        retry_threshold = DEFAULT_MAX_RETRIES

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < retry_threshold:
                raise TimeoutError("Timeout")
            return {
                "message": {
                    "content": to_toml(
                        {
                            "is_task": True,
                            "confidence": 0.9,
                            "description": "Test",
                            "priority": "medium",
                        }
                    )
                }
            }

        with patch("ollama.AsyncClient.chat", side_effect=mock_chat):
            result = await classifier.classify_text("test text")
            assert call_count == retry_threshold
            assert result.is_task is True

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self) -> None:
        """Test that retry uses exponential backoff."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(max_retries=DEFAULT_MAX_RETRIES)

        call_times = []
        retry_threshold = DEFAULT_MAX_RETRIES

        async def mock_chat(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < retry_threshold:
                raise TimeoutError("Timeout")
            return {
                "message": {
                    "content": to_toml(
                        {
                            "is_task": True,
                            "confidence": 0.9,
                            "description": "Test",
                            "priority": "medium",
                            "due_date": None,
                        }
                    )
                }
            }

        with patch("ollama.AsyncClient.chat", side_effect=mock_chat):
            await classifier.classify_text("test text")

            # Verify exponential backoff (each delay should be longer)
            if len(call_times) >= retry_threshold:
                delay1 = call_times[1] - call_times[0]
                delay2 = call_times[2] - call_times[1]
                assert delay2 > delay1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_error(self) -> None:
        """Test that exceeding max retries raises ClassificationError."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(max_retries=2)

        with patch(
            "ollama.AsyncClient.chat",
            side_effect=TimeoutError("Timeout"),
        ):
            with pytest.raises(ClassificationError, match="Max retries exceeded"):
                await classifier.classify_text("test text")

    @pytest.mark.asyncio
    async def test_successful_first_attempt_no_retry(self) -> None:
        """Test that successful first attempt doesn't retry."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(max_retries=DEFAULT_MAX_RETRIES)

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "message": {
                    "content": to_toml(
                        {
                            "is_task": True,
                            "confidence": 0.9,
                            "description": "Test",
                            "priority": "medium",
                            "due_date": None,
                        }
                    )
                }
            }

        with patch("ollama.AsyncClient.chat", side_effect=mock_chat):
            await classifier.classify_text("test text")
            assert call_count == 1


@pytest.mark.unit
class TestTimeoutHandling:
    """Test cases for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_on_slow_response(self) -> None:
        """Test that slow responses trigger timeout."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(timeout=0.1, max_retries=1)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(1.0)
            return {"message": {"content": "{}"}}

        with patch("ollama.AsyncClient.chat", side_effect=slow_response):
            with pytest.raises(ClassificationError):
                await classifier.classify_text("test text")

    @pytest.mark.asyncio
    async def test_timeout_configuration_respected(self) -> None:
        """Test that timeout configuration is respected."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(timeout=CUSTOM_TIMEOUT)
        assert classifier.timeout == CUSTOM_TIMEOUT

    @pytest.mark.asyncio
    async def test_timeout_error_logged(self) -> None:
        """Test that timeout errors are logged."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier(timeout=0.1, max_retries=1)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(1.0)
            return {"message": {"content": "{}"}}

        with patch("ollama.AsyncClient.chat", side_effect=slow_response):
            with patch("logging.Logger.error") as mock_log:
                try:
                    await classifier.classify_text("test text")
                except ClassificationError:
                    pass
                # Verify error was logged
                assert mock_log.called


@pytest.mark.unit
class TestClassificationIntegration:
    """Test cases for end-to-end classification."""

    @pytest.mark.asyncio
    async def test_classify_clear_task(self) -> None:
        """Test classifying clear task text."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        mock_response = {
            "message": {
                "content": to_toml(
                    {
                        "is_task": True,
                        "confidence": 0.95,
                        "description": "Buy groceries tomorrow",
                        "priority": "high",
                        "due_date": (datetime.now() + timedelta(days=1)).isoformat(),
                    }
                )
            }
        }

        with patch("ollama.AsyncClient.chat", return_value=mock_response):
            result = await classifier.classify_text("buy groceries tomorrow")

            assert result.is_task is True
            assert result.confidence == TEST_CONFIDENCE_HIGH
            assert result.description == "Buy groceries tomorrow"
            assert result.priority == TaskPriority.HIGH
            assert result.due_date is not None

    @pytest.mark.asyncio
    async def test_classify_non_task(self) -> None:
        """Test classifying non-task text."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        mock_response = {
            "message": {
                "content": to_toml(
                    {
                        "is_task": False,
                        "confidence": 0.9,
                        "description": None,
                        "priority": None,
                        "due_date": None,
                    }
                )
            }
        }

        with patch("ollama.AsyncClient.chat", return_value=mock_response):
            result = await classifier.classify_text("the weather is nice today")

            assert result.is_task is False
            assert result.confidence == TEST_CONFIDENCE_MEDIUM
            assert result.description is None

    @pytest.mark.asyncio
    async def test_classify_ambiguous_text(self) -> None:
        """Test classifying ambiguous text."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        mock_response = {
            "message": {
                "content": to_toml(
                    {
                        "is_task": False,
                        "confidence": 0.6,
                        "description": None,
                        "priority": None,
                        "due_date": None,
                    }
                )
            }
        }

        with patch("ollama.AsyncClient.chat", return_value=mock_response):
            result = await classifier.classify_text("maybe I should do something")

            assert result.confidence < CONFIDENCE_THRESHOLD  # Below typical threshold

    @pytest.mark.asyncio
    async def test_classify_empty_text(self) -> None:
        """Test classifying empty text."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        with pytest.raises(ClassificationError, match="Empty text"):
            await classifier.classify_text("")

    @pytest.mark.asyncio
    async def test_classify_with_performance_tracking(self) -> None:
        """Test that classification tracks performance metrics."""
        from local_ai.task_management.llm_classifier import LLMClassifier

        classifier = LLMClassifier()

        mock_response = {
            "message": {
                "content": to_toml(
                    {
                        "is_task": True,
                        "confidence": 0.9,
                        "description": "Test task",
                        "priority": "medium",
                        "due_date": None,
                    }
                )
            }
        }

        with patch("ollama.AsyncClient.chat", return_value=mock_response):
            result = await classifier.classify_text("test text")

            # Verify metadata includes timing information
            assert "inference_time" in result.metadata
            assert result.metadata["inference_time"] >= 0
