# Task Management Configuration Guide

This guide covers all configuration options for the Local AI task management system, including examples and integration patterns.

## Configuration Overview

Task management configuration is centralized in `src/local_ai/task_management/config.py`. All settings have sensible defaults optimized for Linux systems with 8GB GPU.

## Configuration Constants

### Task Detection Settings

```python
# Enable/disable task detection globally
TASK_DETECTION_ENABLED = True

# Minimum confidence score to create a task (0.0 - 1.0)
# Lower = more tasks detected (may include false positives)
# Higher = fewer tasks detected (may miss some tasks)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Enable task detection from speech-to-text transcriptions
TASK_DETECTION_FROM_SPEECH_ENABLED = True
```

**Examples:**

```python
# Conservative (fewer false positives)
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Aggressive (catch more tasks)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Disable task detection temporarily
TASK_DETECTION_ENABLED = False
```

### LLM Configuration

```python
# Ollama model for task classification
# Minimum recommended: llama3.2:3b
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Ollama API endpoint
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Request timeout in seconds
DEFAULT_OLLAMA_TIMEOUT = 10.0

# Number of retry attempts on failure
DEFAULT_OLLAMA_MAX_RETRIES = 3

# LLM temperature (0.0 = deterministic, 1.0 = creative)
# Keep low for consistent task detection
DEFAULT_OLLAMA_TEMPERATURE = 0.1
```

**Examples:**

```python
# Use larger model for better accuracy
DEFAULT_OLLAMA_MODEL = "llama3.2:7b"

# Remote Ollama instance
DEFAULT_OLLAMA_BASE_URL = "http://192.168.1.100:11434"

# Faster timeout for quick responses
DEFAULT_OLLAMA_TIMEOUT = 5.0

# More retries for unreliable networks
DEFAULT_OLLAMA_MAX_RETRIES = 5

# Slightly more creative responses
DEFAULT_OLLAMA_TEMPERATURE = 0.2
```

### Storage Configuration

```python
# SQLite database file path
DEFAULT_DATABASE_PATH = os.path.expanduser("~/.local-ai/tasks.db")

# Enable Write-Ahead Logging for better concurrency
DEFAULT_WAL_MODE = True

# Enable automatic database backups
DEFAULT_BACKUP_ENABLED = True

# Database schema version
SCHEMA_VERSION = 1
```

**Examples:**

```python
# Custom database location
DEFAULT_DATABASE_PATH = "/var/lib/local-ai/tasks.db"

# Disable WAL mode (simpler, but slower)
DEFAULT_WAL_MODE = False

# Disable backups (not recommended)
DEFAULT_BACKUP_ENABLED = False
```

### MCP Server Configuration

```python
# Enable MCP server
DEFAULT_MCP_ENABLED = True

# MCP server host
DEFAULT_MCP_HOST = "localhost"

# MCP server port
DEFAULT_MCP_PORT = 3000

# MCP server name for client configuration
DEFAULT_MCP_SERVER_NAME = "local-ai-tasks"
```

**Examples:**

```python
# Bind to all interfaces (use with caution)
DEFAULT_MCP_HOST = "0.0.0.0"

# Use different port
DEFAULT_MCP_PORT = 8080

# Custom server name
DEFAULT_MCP_SERVER_NAME = "my-task-server"

# Disable MCP server
DEFAULT_MCP_ENABLED = False
```

### LLM Prompt Template

The classification prompt is configurable and uses dynamic date placeholders:

```python
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
```

**Placeholders:**

- `{text}` - Input text to analyze
- `{today}` - Current date (YYYY-MM-DD)
- `{day_name}` - Current day name (Monday, Tuesday, etc.)
- `{tomorrow}` - Tomorrow's date
- `{friday}` - Next Friday's date
- `{monday}` - Next Monday's date
- `{next_week}` - Date one week from today

**Custom Prompt Example:**

```python
# Simpler prompt for faster processing
DEFAULT_CLASSIFICATION_PROMPT = """Is this a task? Answer in JSON:
{{"is_task": true/false, "confidence": 0.0-1.0, "description": "task summary", "priority": "low/medium/high", "due_date": "YYYY-MM-DD or null"}}

Today is {today}. Analyze: "{text}"
"""
```

## Environment Variables

Override configuration at runtime using environment variables:

```bash
# Ollama configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2:3b"
export OLLAMA_TIMEOUT="10"

# Database configuration
export TASK_DB_PATH="$HOME/.local-ai/tasks.db"

# Task detection
export TASK_DETECTION_ENABLED="true"
export CONFIDENCE_THRESHOLD="0.7"

# MCP server
export MCP_ENABLED="true"
export MCP_HOST="localhost"
export MCP_PORT="3000"
```

## Integration Patterns

### Pattern 1: Automatic Speech Integration

Task detection happens automatically when speech-to-text produces transcriptions.

**Configuration:**

```python
# In config.py
TASK_DETECTION_FROM_SPEECH_ENABLED = True
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
```

**Usage:**

```bash
# Just run the application
local-ai

# Speak: "Remind me to call mom tomorrow"
# Task is automatically detected and stored
```

**Code (automatic in SpeechToTextService):**

```python
from local_ai.speech_to_text.service import SpeechToTextService

async def main():
    service = SpeechToTextService()
    # Task detection is automatic when enabled
    await service.start_listening()
```

### Pattern 2: Manual Text Processing

Process text directly without speech-to-text.

**Configuration:**

```python
# In config.py
TASK_DETECTION_ENABLED = True
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
```

**Usage:**

```python
import asyncio
from local_ai.task_management.task_detection_service import TaskDetectionService
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager
from local_ai.task_management.llm_classifier import LLMClassifier

async def process_text(text: str):
    # Initialize components
    db = TaskDatabase()
    await db.initialize()

    classifier = LLMClassifier()
    manager = TaskListManager(db)
    service = TaskDetectionService(classifier, manager)

    # Detect task
    result = await service.detect_task_from_text(text)

    if result.task_detected:
        print(f"Task: {result.task.description}")
        print(f"Priority: {result.task.priority}")
        print(f"Due: {result.task.due_date}")
    else:
        print("No task detected")

    await db.close()

# Process text
asyncio.run(process_text("Buy groceries tomorrow"))
```

### Pattern 3: Direct Task Management

Create and manage tasks directly without LLM classification.

**Usage:**

```python
import asyncio
from datetime import datetime, timedelta
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager
from local_ai.task_management.models import TaskPriority, TaskStatus

async def manage_tasks():
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

    # List all tasks
    tasks = await manager.list_tasks(status=TaskStatus.PENDING)
    for t in tasks:
        print(f"- [{t.status}] {t.description}")

    # Update task status
    await manager.update_task_status(task.id, TaskStatus.COMPLETED)

    # Get statistics
    stats = await manager.get_statistics()
    print(f"Total: {stats['total']}, Completed: {stats['completed']}")

    await db.close()

asyncio.run(manage_tasks())
```

### Pattern 4: MCP Client Integration

Access tasks from AI assistants via Model Context Protocol.

**MCP Client Configuration (Claude Desktop on Linux):**

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

**MCP Client Configuration (Cursor):**

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

**Using MCP Tools:**

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

### Pattern 5: Custom Integration

Build custom integrations using the task management API.

**Example: CLI Tool**

```python
#!/usr/bin/env python3
import asyncio
import sys
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager

async def cli_main():
    db = TaskDatabase()
    await db.initialize()
    manager = TaskListManager(db)

    command = sys.argv[1] if len(sys.argv) > 1 else "list"

    if command == "list":
        tasks = await manager.list_tasks()
        for task in tasks:
            print(f"[{task.status}] {task.description}")

    elif command == "add":
        description = " ".join(sys.argv[2:])
        task = await manager.add_task(description=description)
        print(f"Added: {task.id}")

    elif command == "complete":
        task_id = sys.argv[2]
        await manager.update_task_status(task_id, "completed")
        print(f"Completed: {task_id}")

    await db.close()

if __name__ == "__main__":
    asyncio.run(cli_main())
```

**Usage:**

```bash
# List tasks
./task_cli.py list

# Add task
./task_cli.py add "Buy groceries"

# Complete task
./task_cli.py complete abc-123-def
```

**Example: Web API**

```python
from fastapi import FastAPI
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager

app = FastAPI()
db = TaskDatabase()
manager = TaskListManager(db)

@app.on_event("startup")
async def startup():
    await db.initialize()

@app.get("/tasks")
async def list_tasks():
    tasks = await manager.list_tasks()
    return {"tasks": [t.dict() for t in tasks]}

@app.post("/tasks")
async def create_task(description: str, priority: str = "medium"):
    task = await manager.add_task(description=description, priority=priority)
    return {"task": task.dict()}

@app.patch("/tasks/{task_id}")
async def update_task(task_id: str, status: str):
    await manager.update_task_status(task_id, status)
    return {"status": "updated"}
```

## Performance Tuning

### For CPU-Only Systems

```python
# Use smaller model
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"  # Faster but less accurate

# Reduce timeout
DEFAULT_OLLAMA_TIMEOUT = 5.0

# Fewer retries
DEFAULT_OLLAMA_MAX_RETRIES = 1
```

### For GPU Systems

```python
# Use larger model for better accuracy
DEFAULT_OLLAMA_MODEL = "llama3.2:7b"

# Longer timeout for complex processing
DEFAULT_OLLAMA_TIMEOUT = 15.0

# More retries for reliability
DEFAULT_OLLAMA_MAX_RETRIES = 5
```

### For High-Volume Processing

```python
# Enable WAL mode for better concurrency
DEFAULT_WAL_MODE = True

# Increase database cache (in database.py)
await self.db.execute("PRAGMA cache_size = -64000")  # 64MB

# Use connection pooling (future enhancement)
```

## Troubleshooting Configuration

### Check Current Configuration

```python
from local_ai.task_management import config

print(f"Model: {config.DEFAULT_OLLAMA_MODEL}")
print(f"Base URL: {config.DEFAULT_OLLAMA_BASE_URL}")
print(f"Confidence: {config.DEFAULT_CONFIDENCE_THRESHOLD}")
print(f"Database: {config.DEFAULT_DATABASE_PATH}")
```

### Test Ollama Connection

```python
import asyncio
from local_ai.task_management.llm_classifier import LLMClassifier

async def test_ollama():
    classifier = LLMClassifier()
    result = await classifier.classify_text("Buy groceries tomorrow")
    print(f"Classification: {result}")

asyncio.run(test_ollama())
```

### Validate Database

```bash
# Check if database exists
ls -lh ~/.local-ai/tasks.db

# Check database integrity
sqlite3 ~/.local-ai/tasks.db "PRAGMA integrity_check;"

# View schema
sqlite3 ~/.local-ai/tasks.db ".schema"

# Count tasks
sqlite3 ~/.local-ai/tasks.db "SELECT COUNT(*) FROM tasks;"
```

### Test MCP Server

```bash
# Start server manually
python -m local_ai.task_management.mcp_server --verbose

# Test with curl (if HTTP endpoint available)
curl http://localhost:3000/health
```

## Configuration Best Practices

### Development

```python
# Lower confidence for testing
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use test database
DEFAULT_DATABASE_PATH = "/tmp/tasks_test.db"
```

### Production

```python
# Higher confidence for accuracy
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Production database location
DEFAULT_DATABASE_PATH = "/var/lib/local-ai/tasks.db"

# Enable backups
DEFAULT_BACKUP_ENABLED = True

# Reasonable timeout
DEFAULT_OLLAMA_TIMEOUT = 10.0
```

### Testing

```python
# Disable task detection during tests
TASK_DETECTION_ENABLED = False

# Use in-memory database
DEFAULT_DATABASE_PATH = ":memory:"

# Mock Ollama responses
# (Use pytest fixtures)
```

## Security Considerations

### Network Security

```python
# Bind to localhost only
DEFAULT_MCP_HOST = "localhost"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Never expose to public internet without authentication
```

### Database Security

```bash
# Set proper file permissions
chmod 600 ~/.local-ai/tasks.db

# Restrict directory access
chmod 700 ~/.local-ai/
```

### Input Validation

```python
# Input sanitization is built-in
# Max text length: 500 characters
# Quote escaping: automatic
# Newline removal: automatic
```

## Migration and Upgrades

### Changing Database Location

```bash
# Stop application
pkill local-ai

# Move database
mv ~/.local-ai/tasks.db /new/location/tasks.db

# Update configuration
export TASK_DB_PATH="/new/location/tasks.db"

# Restart application
local-ai
```

### Changing Ollama Model

```bash
# Pull new model
ollama pull llama3.2:7b

# Update configuration
export OLLAMA_MODEL="llama3.2:7b"

# Restart application
local-ai
```

### Schema Migrations

```python
# Schema version is tracked automatically
# Future migrations will be handled by database.py
# Current version: 1
```

## Additional Resources

- Main README: `README.md`
- Deployment Guide: `docs/task-management-deployment.md`
- API Documentation: `src/local_ai/task_management/`
- Test Examples: `tests/test_task_management/`
