# Task Management Guide

Complete guide to using the Local AI task management system with voice integration.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Task Detection Examples](#task-detection-examples)
- [MCP Integration](#mcp-integration)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The task management system automatically detects tasks from text input (including voice transcriptions) and stores them in a local database. Tasks are exposed via Model Context Protocol (MCP) for integration with AI tools like Claude Desktop and Cursor.

**Key Benefits:**

- **Privacy-first**: All processing happens locally using Ollama
- **Voice-enabled**: Automatically detects tasks from speech-to-text
- **AI-powered**: Uses llama3.2:3b for intelligent task classification
- **Persistent**: SQLite database with full history tracking
- **Integrated**: MCP server for seamless AI assistant integration

## Features

- **Automatic Detection**: AI analyzes text to identify actionable tasks
- **Smart Extraction**: Extracts task description, priority, and due dates
- **Local Processing**: Uses Ollama with llama3.2:3b model (runs offline)
- **Persistent Storage**: SQLite database with full history tracking
- **MCP Integration**: Expose tasks to Claude, Cursor, and other MCP clients
- **Voice Integration**: Automatically processes speech-to-text output

## Quick Start

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required model (2GB download)
ollama pull llama3.2:3b

# Start Ollama service
ollama serve
```

### 2. Install Local AI

```bash
# Clone and install
git clone https://github.com/Djeisen642/local-ai-python.git
cd local-ai-python
uv pip install -e .
```

### 3. Use with Voice

```bash
# Start the application
local-ai

# Speak into your microphone:
# "Remind me to finish the report by Friday"

# Task is automatically detected and stored
```

### 4. Access via MCP

Configure your MCP client (see [MCP Integration](#mcp-integration) below).

## Task Detection Examples

The system recognizes various task patterns from natural language:

| Input                           | Detected Task | Priority | Due Date    |
| ------------------------------- | ------------- | -------- | ----------- |
| "Finish the report by Friday"   | Finish report | High     | This Friday |
| "I should check my email later" | Check email   | Low      | None        |
| "Call mom tomorrow"             | Call mom      | Medium   | Tomorrow    |
| "Meeting at 3pm next Monday"    | Meeting       | Medium   | Next Monday |
| "Buy groceries this weekend"    | Buy groceries | Medium   | This Sat    |
| "I went to the store"           | _(no task)_   | -        | -           |

**Detection Confidence:**

- **High (0.8-1.0)**: Clear task with action verb and context
- **Medium (0.6-0.8)**: Likely task but ambiguous
- **Low (0.0-0.6)**: Probably not a task (filtered out by default)

## MCP Integration

### Available Tools

The MCP server exposes these tools for AI assistants:

- **`list_tasks`** - List all tasks with optional filters (status, priority)
- **`add_task`** - Manually create a new task
- **`update_task_status`** - Mark tasks as completed, in progress, etc.
- **`delete_task`** - Remove a task
- **`get_task_statistics`** - Get task counts and statistics

### Configure Claude Desktop

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-ai-tasks": {
      "command": "python",
      "args": ["-m", "local_ai.task_management.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/local-ai-python/src"
      }
    }
  }
}
```

### Configure Cursor

Edit `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "local-ai-tasks": {
      "command": "python",
      "args": ["-m", "local_ai.task_management.mcp_server"]
    }
  }
}
```

### Using MCP Tools

In your AI assistant (Claude, Cursor, etc.):

```
User: "List all my pending tasks"
Assistant: [calls list_tasks tool with status=pending]

User: "Add a task to review the code by Friday"
Assistant: [calls add_task tool with description, priority, due_date]

User: "Mark task abc-123 as completed"
Assistant: [calls update_task_status tool]

User: "How many tasks do I have?"
Assistant: [calls get_task_statistics tool]
```

## Python API

### Automatic Task Detection from Text

```python
import asyncio
from local_ai.task_management.task_detection_service import TaskDetectionService
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager
from local_ai.task_management.llm_classifier import LLMClassifier

async def main():
    # Initialize components
    db = TaskDatabase()
    await db.initialize()

    classifier = LLMClassifier()
    manager = TaskListManager(db)
    service = TaskDetectionService(classifier, manager)

    # Detect task from text
    result = await service.detect_task_from_text(
        "Finish the report by Friday"
    )

    if result.task_detected:
        print(f"Task created: {result.task.description}")
        print(f"Priority: {result.task.priority}")
        print(f"Due: {result.task.due_date}")
        print(f"Confidence: {result.confidence:.1%}")

    await db.close()

asyncio.run(main())
```

### Direct Task Management

```python
import asyncio
from datetime import datetime, timedelta
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager
from local_ai.task_management.models import TaskPriority, TaskStatus

async def main():
    db = TaskDatabase()
    await db.initialize()
    manager = TaskListManager(db)

    # Add task manually
    task = await manager.add_task(
        description="Review pull request",
        priority=TaskPriority.HIGH,
        due_date=datetime.now() + timedelta(days=1),
        source="manual"
    )
    print(f"Created task: {task.id}")

    # List all pending tasks
    tasks = await manager.list_tasks(status=TaskStatus.PENDING)
    for t in tasks:
        print(f"- [{t.status}] {t.description}")

    # Update task status
    await manager.update_task_status(task.id, TaskStatus.COMPLETED)

    # Get statistics
    stats = await manager.get_statistics()
    print(f"Total: {stats['total']}, Completed: {stats['completed']}")

    await db.close()

asyncio.run(main())
```

### Automatic from Speech-to-Text

```python
import asyncio
from local_ai.speech_to_text.service import SpeechToTextService

async def main():
    # Task detection happens automatically when enabled
    service = SpeechToTextService()
    await service.start_listening()

    # Speak: "Remind me to call mom tomorrow"
    # Task is automatically detected and stored

    await asyncio.sleep(30)  # Listen for 30 seconds
    await service.stop_listening()

asyncio.run(main())
```

## Configuration

### Basic Settings

Edit `src/local_ai/task_management/config.py`:

```python
# Task Detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to create task
TASK_DETECTION_ENABLED = True       # Enable/disable task detection

# LLM Configuration
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 10.0
DEFAULT_OLLAMA_MAX_RETRIES = 3

# Storage
DEFAULT_DATABASE_PATH = "~/.local-ai/tasks.db"

# MCP Server
DEFAULT_MCP_ENABLED = True
DEFAULT_MCP_HOST = "localhost"
DEFAULT_MCP_PORT = 3000
```

### Environment Variables

Override configuration at runtime:

```bash
# Ollama configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2:3b"

# Database configuration
export TASK_DB_PATH="$HOME/.local-ai/tasks.db"

# Task detection
export TASK_DETECTION_ENABLED="true"
export CONFIDENCE_THRESHOLD="0.7"
```

### Tuning Confidence Threshold

```python
# Conservative (fewer false positives)
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Aggressive (catch more tasks)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Default (balanced)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
```

For detailed configuration options, see [Configuration Guide](task-management-configuration.md).

## Troubleshooting

### Ollama Connection Issues

**"Ollama connection failed" or "Model not found"**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
sudo systemctl start ollama

# Pull the required model
ollama pull llama3.2:3b

# Verify model is available
ollama list
```

### Tasks Not Being Detected

**From speech:**

- Check that `TASK_DETECTION_FROM_SPEECH_ENABLED = True` in config
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check confidence threshold (default 0.7) - lower if needed
- Review logs with `--verbose` flag to see classification results

**From text:**

```python
# Test classification directly
from local_ai.task_management.llm_classifier import LLMClassifier

classifier = LLMClassifier()
result = await classifier.classify_text("Buy groceries tomorrow")
print(result)
```

### MCP Server Issues

**Server not connecting:**

```bash
# Check if port 3000 is available
netstat -tuln | grep 3000

# Start server manually for debugging
python -m local_ai.task_management.mcp_server --verbose

# Check MCP client configuration
cat ~/.config/Claude/claude_desktop_config.json
```

### Database Issues

**Database errors or corruption:**

```bash
# Backup existing database
cp ~/.local-ai/tasks.db ~/.local-ai/tasks.db.backup

# Check database integrity
sqlite3 ~/.local-ai/tasks.db "PRAGMA integrity_check;"

# Reset database (will lose tasks)
rm ~/.local-ai/tasks.db
# Database will be recreated on next run
```

### Low Detection Accuracy

**Improve task detection:**

- Ensure you're using llama3.2:3b (not smaller models)
- Speak clearly and use explicit task language:
  - ✅ "I need to finish the report"
  - ✅ "Remind me to call mom"
  - ❌ "The report thing"
  - ❌ "Maybe later"
- Check classification confidence in logs (`--verbose`)
- Consider adjusting `DEFAULT_CONFIDENCE_THRESHOLD` in config

### Performance Issues

**Slow task detection:**

```python
# Use smaller model (faster but less accurate)
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"

# Reduce timeout
DEFAULT_OLLAMA_TIMEOUT = 5.0

# Fewer retries
DEFAULT_OLLAMA_MAX_RETRIES = 1
```

## Additional Resources

- [Deployment Guide](task-management-deployment.md) - Complete deployment instructions
- [Configuration Guide](task-management-configuration.md) - Detailed configuration options
- [Main README](../README.md) - Project overview and speech-to-text documentation
- [Ollama Documentation](https://ollama.com/docs) - Ollama setup and usage
