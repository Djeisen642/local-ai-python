"""Configuration constants for task management functionality."""

import os

# Task Detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
TASK_DETECTION_ENABLED = True

# LLM Configuration
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 10.0  # seconds
DEFAULT_OLLAMA_MAX_RETRIES = 3
DEFAULT_OLLAMA_TEMPERATURE = 0.1

# Storage Configuration
DEFAULT_DATABASE_PATH = os.path.expanduser("~/.local-ai/tasks.db")
DEFAULT_WAL_MODE = True
DEFAULT_BACKUP_ENABLED = True

# MCP Server Configuration
DEFAULT_MCP_ENABLED = True
DEFAULT_MCP_HOST = "localhost"
DEFAULT_MCP_PORT = 3000
DEFAULT_MCP_SERVER_NAME = "local-ai-tasks"

# Database Schema Version
SCHEMA_VERSION = 1

# Speech-to-Text Integration
TASK_DETECTION_FROM_SPEECH_ENABLED = True  # Enable task detection from transcriptions

# LLM Prompt Template
# Placeholders: {text}, {today}, {day_name}, {tomorrow}, {friday}, {monday}, {next_week}
DEFAULT_CLASSIFICATION_PROMPT = """Decide if the text is a task or just a statement.

Output this JSON only:
{{"is_task": true/false, "confidence": 0.0-1.0, "description": "short actionable summary or null", "priority": "low/medium/high or null", "due_date": "YYYY-MM-DD or null"}}

Rules:
- Today is {today} ({day_name})
- Date conversions: "tomorrow"={tomorrow}, "Friday"={friday}, "Monday"={monday}, "next week"={next_week}
- Confidence: ~0.9 for clear tasks, ~0.5 for unclear, ~0.1 for statements
- Priority: "high" if urgent/deadline, "medium" for normal tasks, "low" for optional

Examples:
Input: "Please finish the report by Friday"
Output: {{"is_task": true, "confidence": 0.9, "description": "Finish report", "priority": "high", "due_date": "{friday}"}}

Input: "I went to the store"
Output: {{"is_task": false, "confidence": 0.95, "description": null, "priority": null, "due_date": null}}

Input: "Maybe check the email later"
Output: {{"is_task": true, "confidence": 0.6, "description": "Check email", "priority": "low", "due_date": null}}

Now analyze this:
"{text}"
"""
