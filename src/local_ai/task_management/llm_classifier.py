"""LLM classifier for task detection using Ollama."""

import asyncio
import logging
import time
import tomllib
from datetime import datetime
from typing import Any

import ollama

from .config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MAX_RETRIES,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_TIMEOUT,
)
from .exceptions import ClassificationError
from .models import ClassificationResult, TaskPriority

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Classifies text using local LLM via Ollama to detect tasks."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        max_retries: int = DEFAULT_OLLAMA_MAX_RETRIES,
        temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    ) -> None:
        """
        Initialize LLM classifier.

        Args:
            model: Ollama model name
            base_url: Ollama service URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            temperature: LLM temperature for generation
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = ollama.AsyncClient(host=base_url)

    def _generate_prompt(self, text: str) -> str:
        """
        Generate classification prompt for the LLM.

        Args:
            text: Input text to classify

        Returns:
            Formatted prompt string
        """
        # Sanitize input to prevent prompt injection
        sanitized = text.replace("\n", " ").replace('"', "'")[:500]

        return f"""Is this an actionable task? Respond ONLY in TOML format.

TEXT: {sanitized}

TOML format:
is_task = true/false
confidence = 0.0-1.0
description = "text" or null
priority = "low/medium/high" or null
due_date = "ISO8601" or null"""

    def _parse_response(self, response_text: str) -> ClassificationResult:
        """
        Parse LLM response into ClassificationResult.

        Args:
            response_text: Raw TOML response text from LLM

        Returns:
            ClassificationResult object

        Raises:
            ClassificationError: If response cannot be parsed
        """
        try:
            data = tomllib.loads(response_text)
        except tomllib.TOMLDecodeError as e:
            raise ClassificationError(f"Invalid TOML response: {e}") from e

        # Validate required fields
        if "is_task" not in data:
            raise ClassificationError("Missing required field: is_task")
        if "confidence" not in data:
            raise ClassificationError("Missing required field: confidence")

        # Parse priority
        priority = None
        if data.get("priority"):
            priority_str = data["priority"].lower()
            try:
                priority = TaskPriority(priority_str)
            except ValueError:
                logger.warning(
                    f"Invalid priority '{data['priority']}', defaulting to medium"
                )
                priority = TaskPriority.MEDIUM

        # Parse due date
        due_date = None
        if data.get("due_date"):
            try:
                due_date = datetime.fromisoformat(data["due_date"])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid due date '{data['due_date']}': {e}")
                due_date = None

        # Extract metadata (any extra fields)
        metadata: dict[str, Any] = {}
        known_fields = {
            "is_task",
            "confidence",
            "description",
            "priority",
            "due_date",
        }
        for key, value in data.items():
            if key not in known_fields:
                metadata[key] = value

        return ClassificationResult(
            is_task=data["is_task"],
            confidence=data["confidence"],
            description=data.get("description"),
            priority=priority,
            due_date=due_date,
            metadata=metadata,
        )

    async def classify_text(self, text: str) -> ClassificationResult:
        """
        Classify text to determine if it represents a task.

        Args:
            text: Input text to classify

        Returns:
            ClassificationResult with classification details

        Raises:
            ClassificationError: If classification fails
        """
        if not text or not text.strip():
            raise ClassificationError("Empty text provided for classification")

        prompt = self._generate_prompt(text)
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                # Make async call to Ollama
                response = await asyncio.wait_for(
                    self._client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": self.temperature},
                    ),
                    timeout=self.timeout,
                )

                # Extract response content
                content = response["message"]["content"]

                # Parse response
                result = self._parse_response(content)

                # Add performance metadata
                inference_time = time.time() - start_time
                result.metadata["inference_time"] = inference_time

                logger.info(
                    f"Classification complete: is_task={result.is_task}, "
                    f"confidence={result.confidence:.2f}, "
                    f"time={inference_time:.3f}s"
                )

                return result

            except TimeoutError as e:
                logger.error(f"Timeout on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise ClassificationError(
                        f"Max retries exceeded after {self.max_retries} attempts"
                    ) from e

            except ConnectionError as e:
                logger.error(f"Connection error: {e}")
                raise ClassificationError(f"Connection failed: {e}") from e

            except Exception as e:
                logger.error(f"Classification error: {e}")
                raise ClassificationError(f"Classification failed: {e}") from e

        raise ClassificationError(
            f"Max retries exceeded after {self.max_retries} attempts"
        )
