"""LLM classifier for task detection using Ollama."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import ollama

from .config import (
    DEFAULT_CLASSIFICATION_PROMPT,
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
        prompt_template: str | None = None,
    ) -> None:
        """
        Initialize LLM classifier.

        Args:
            model: Ollama model name
            base_url: Ollama service URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            temperature: LLM temperature for generation
            prompt_template: Optional custom prompt template (must contain {text})
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.prompt_template = prompt_template
        self._client = ollama.AsyncClient(host=base_url)

    def _generate_prompt(self, text: str) -> str:
        """
        Generate classification prompt for the LLM.

        Args:
            text: Input text to classify

        Returns:
            Formatted prompt string with dynamic date values
        """
        # Sanitize input to prevent prompt injection
        sanitized = text.replace("\n", " ").replace('"', "'")[:500]

        # Use custom prompt template if provided, otherwise use config default
        prompt_template = self.prompt_template or DEFAULT_CLASSIFICATION_PROMPT

        # Calculate dynamic date values
        today = datetime.now()
        tomorrow = today + timedelta(days=1)

        # Calculate next Friday
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        next_friday = today + timedelta(days=days_until_friday)

        # Calculate next Monday
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        next_monday = today + timedelta(days=days_until_monday)

        # Calculate next week (7 days from today)
        next_week = today + timedelta(days=7)

        # Format prompt with all dynamic values
        prompt = prompt_template.format(
            text=sanitized,
            today=today.strftime("%Y-%m-%d"),
            day_name=today.strftime("%A"),
            tomorrow=tomorrow.strftime("%Y-%m-%d"),
            friday=next_friday.strftime("%Y-%m-%d"),
            monday=next_monday.strftime("%Y-%m-%d"),
            next_week=next_week.strftime("%Y-%m-%d"),
        )

        return prompt

    def _parse_response(self, response_text: str) -> ClassificationResult:
        """
        Parse LLM response into ClassificationResult.

        Args:
            response_text: Raw JSON response text from LLM

        Returns:
            ClassificationResult object

        Raises:
            ClassificationError: If response cannot be parsed
        """
        # Clean up common LLM response issues
        cleaned = response_text.strip()

        # Remove markdown code blocks if present
        if "```" in cleaned:
            lines = cleaned.split("\n")
            json_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or (not in_code and "{" in line):
                    json_lines.append(line)
            cleaned = "\n".join(json_lines)

        # Find JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = cleaned[start:end]
        else:
            raise ClassificationError("No JSON object found in response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Log the problematic response for debugging
            logger.debug(f"Failed to parse JSON response: {json_str}")
            raise ClassificationError(f"Invalid JSON response: {e}") from e

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

        logger.info(f"🔍 Classifying text: '{text}'")
        logger.debug(f"📝 Generated prompt:\n{prompt}")

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
                logger.debug(f"🤖 LLM raw response:\n{content}")

                # Parse response
                result = self._parse_response(content)

                # Add performance metadata
                inference_time = time.time() - start_time
                result.metadata["inference_time"] = inference_time

                logger.info(
                    f"✅ Classification complete: is_task={result.is_task}, "
                    f"confidence={result.confidence:.2f}, "
                    f"description='{result.description}', "
                    f"priority={result.priority}, "
                    f"due_date={result.due_date}, "
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
