"""Configuration constants for task management functionality."""

import os

# Task Detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
TASK_DETECTION_ENABLED = True

# LLM Configuration
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
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
