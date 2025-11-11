# Task Management Deployment Guide

This guide covers the complete deployment process for the Local AI task management system, including Ollama installation, model setup, and MCP server configuration.

## Overview

The task management system consists of:

- **Ollama**: Local LLM inference server
- **llama3.2:3b**: Language model for task classification
- **SQLite Database**: Persistent task storage
- **MCP Server**: Model Context Protocol integration

## Prerequisites

- Python 3.13+
- Linux (Ubuntu 20.04+, Debian 11+, or similar)
- 8GB RAM minimum (16GB recommended)
- 4GB disk space for model

## Step 1: Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

**Docker Alternative:**

```bash
# Run Ollama in Docker
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama

# Verify
curl http://localhost:11434/api/tags
```

## Step 2: Start Ollama Service

Ollama installs as a systemd service automatically:

```bash
# Check service status
sudo systemctl status ollama

# Start if not running
sudo systemctl start ollama

# Enable on boot
sudo systemctl enable ollama
```

**Manual Start (for debugging):**

```bash
# Run in foreground
ollama serve

# Or run in background
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

## Step 3: Download llama3.2:3b Model

### Pull the Model

```bash
# Download llama3.2:3b (~2GB)
ollama pull llama3.2:3b

# This will take 5-10 minutes depending on connection speed
```

### Verify Model Installation

```bash
# List installed models
ollama list

# Expected output:
# NAME              ID              SIZE      MODIFIED
# llama3.2:3b       a80c4f17acd5    2.0 GB    X minutes ago

# Test the model
ollama run llama3.2:3b "Hello, how are you?"
```

### Model Selection Notes

**Why llama3.2:3b?**

- **Minimum recommended**: Smaller models (1b) have poor task detection accuracy
- **Size**: 2GB download, ~3GB RAM usage
- **Performance**: 500ms-1s inference on 8GB GPU, 2-3s on CPU
- **Accuracy**: 70-85% task detection accuracy in testing

**Alternative Models:**

```bash
# Smaller (NOT recommended - low accuracy)
ollama pull llama3.2:1b  # 1GB, faster but unreliable

# Larger (better accuracy, slower)
ollama pull llama3.2:7b  # 4GB, 90%+ accuracy
ollama pull llama3.1:8b  # 5GB, best accuracy
```

## Step 4: Install Local AI Package

```bash
# Clone repository
git clone https://github.com/Djeisen642/local-ai-python.git
cd local-ai-python

# Install with task management dependencies
uv pip install -e .

# Verify installation
local-ai --help
```

## Step 5: Configure Task Management

### Default Configuration

The system works out-of-the-box with defaults:

```python
# src/local_ai/task_management/config.py

# Ollama connection
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 10.0

# Database location
DEFAULT_DATABASE_PATH = "~/.local-ai/tasks.db"

# MCP server
DEFAULT_MCP_HOST = "localhost"
DEFAULT_MCP_PORT = 3000
```

### Custom Configuration

Create a custom config file (optional):

```python
# config_override.py
import os
os.environ['OLLAMA_BASE_URL'] = 'http://custom-host:11434'
os.environ['TASK_DB_PATH'] = '/custom/path/tasks.db'
```

### Environment Variables

```bash
# Override Ollama connection
export OLLAMA_BASE_URL="http://localhost:11434"

# Override database path
export TASK_DB_PATH="$HOME/.local-ai/tasks.db"

# Disable task detection
export TASK_DETECTION_ENABLED="false"
```

## Step 6: Test the Installation

### Test Ollama Connection

```bash
# Test API endpoint
curl http://localhost:11434/api/tags

# Test model inference
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Is this a task: Buy groceries tomorrow",
  "stream": false
}'
```

### Test Task Detection

```python
# test_task_detection.py
import asyncio
from local_ai.task_management.task_detection_service import TaskDetectionService
from local_ai.task_management.database import TaskDatabase
from local_ai.task_management.task_list_manager import TaskListManager
from local_ai.task_management.llm_classifier import LLMClassifier

async def test():
    db = TaskDatabase()
    await db.initialize()

    classifier = LLMClassifier()
    manager = TaskListManager(db)
    service = TaskDetectionService(classifier, manager)

    # Test task detection
    result = await service.detect_task_from_text(
        "Finish the report by Friday"
    )

    print(f"Task detected: {result.task_detected}")
    if result.task_detected:
        print(f"Description: {result.task.description}")
        print(f"Priority: {result.task.priority}")
        print(f"Confidence: {result.confidence:.2%}")

    await db.close()

asyncio.run(test())
```

Run the test:

```bash
python test_task_detection.py
```

Expected output:

```
Task detected: True
Description: Finish report
Priority: high
Confidence: 89%
```

### Test with Speech-to-Text

```bash
# Start the application
local-ai --verbose

# Speak into microphone:
# "Remind me to call mom tomorrow"

# Check logs for task detection:
# [INFO] Task detected: Call mom (confidence: 0.85)
# [INFO] Task stored with ID: abc-123-def
```

## Step 7: Set Up MCP Server

### Start MCP Server

```bash
# Start as standalone service
python -m local_ai.task_management.mcp_server

# Or with custom port
python -m local_ai.task_management.mcp_server --port 3001
```

### Configure MCP Client

#### Claude Desktop (Linux)

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

#### Cursor IDE

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

### Test MCP Connection

```bash
# Install MCP inspector
npm install -g @modelcontextprotocol/inspector

# Test the server
mcp-inspector python -m local_ai.task_management.mcp_server

# Should show available tools:
# - list_tasks
# - add_task
# - update_task_status
# - delete_task
# - get_task_statistics
```

## Step 8: Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/local-ai-tasks.service`:

```ini
[Unit]
Description=Local AI Task Management
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/local-ai-python
Environment="PATH=/home/your-username/.local/bin:/usr/bin"
ExecStart=/home/your-username/.local/bin/local-ai
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable local-ai-tasks
sudo systemctl start local-ai-tasks
sudo systemctl status local-ai-tasks
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  local-ai:
    build: .
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - TASK_DB_PATH=/data/tasks.db
    volumes:
      - task_data:/data
    restart: unless-stopped

volumes:
  ollama_data:
  task_data:
```

Start services:

```bash
docker-compose up -d
```

### Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Start application
pm2 start local-ai --name local-ai-tasks

# Save configuration
pm2 save

# Enable startup script
pm2 startup
```

## Troubleshooting

### Ollama Not Starting

```bash
# Check logs
journalctl -u ollama -f

# Check port availability
netstat -tuln | grep 11434

# Kill existing process
pkill ollama
ollama serve
```

### Model Download Fails

```bash
# Check disk space
df -h

# Check network connectivity
curl -I https://ollama.com

# Manual download (if needed)
wget https://ollama.com/library/llama3.2:3b
ollama create llama3.2:3b -f ./llama3.2:3b
```

### Database Initialization Fails

```bash
# Check directory permissions
ls -la ~/.local-ai/

# Create directory manually
mkdir -p ~/.local-ai
chmod 755 ~/.local-ai

# Check SQLite version
sqlite3 --version  # Should be 3.35+
```

### MCP Server Connection Issues

```bash
# Check if server is running
curl http://localhost:3000/health

# Check firewall
sudo ufw allow 3000

# Test with different port
python -m local_ai.task_management.mcp_server --port 3001
```

## Performance Tuning

### GPU Acceleration

```bash
# Verify GPU is available to Ollama
nvidia-smi

# Set GPU layers (in Ollama)
ollama run llama3.2:3b --gpu-layers 32
```

### CPU-Only Optimization

```bash
# Reduce concurrent requests
export OLLAMA_NUM_PARALLEL=1

# Reduce context size
export OLLAMA_MAX_LOADED_MODELS=1
```

### Database Optimization

```python
# Enable WAL mode (default)
DEFAULT_WAL_MODE = True

# Increase cache size
# In database.py, add:
await self.db.execute("PRAGMA cache_size = -64000")  # 64MB
```

## Monitoring

### Check System Status

```bash
# Ollama status
curl http://localhost:11434/api/tags

# Database size
du -h ~/.local-ai/tasks.db

# Task count
sqlite3 ~/.local-ai/tasks.db "SELECT COUNT(*) FROM tasks;"
```

### Logs

```bash
# Application logs (with verbose)
local-ai --verbose 2>&1 | tee local-ai.log

# Ollama logs
journalctl -u ollama -f

# MCP server logs
python -m local_ai.task_management.mcp_server --verbose
```

## Security Considerations

### Network Security

```bash
# Bind Ollama to localhost only (default)
# Edit /etc/systemd/system/ollama.service
Environment="OLLAMA_HOST=127.0.0.1:11434"

# Use firewall to restrict access
sudo ufw deny 11434
sudo ufw allow from 127.0.0.1 to any port 11434
```

### Database Security

```bash
# Set proper permissions
chmod 600 ~/.local-ai/tasks.db

# Encrypt database (optional)
# Use SQLCipher instead of SQLite
```

### MCP Authentication

```python
# Add authentication to MCP server (future enhancement)
# Currently MCP runs locally without auth
```

## Backup and Recovery

### Backup Database

```bash
# Manual backup
cp ~/.local-ai/tasks.db ~/.local-ai/tasks.db.backup

# Automated backup (cron)
0 2 * * * cp ~/.local-ai/tasks.db ~/.local-ai/tasks.db.$(date +\%Y\%m\%d)
```

### Restore Database

```bash
# Restore from backup
cp ~/.local-ai/tasks.db.backup ~/.local-ai/tasks.db

# Verify integrity
sqlite3 ~/.local-ai/tasks.db "PRAGMA integrity_check;"
```

## Upgrading

### Update Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Verify version
ollama --version
```

### Update Model

```bash
# Pull latest version
ollama pull llama3.2:3b

# Remove old version (optional)
ollama rm llama3.2:3b@old-version
```

### Update Local AI

```bash
cd local-ai-python
git pull
uv pip install -e .
```

## Next Steps

- Configure MCP client integration
- Set up automated backups
- Monitor task detection accuracy
- Tune confidence threshold for your use case
- Explore integration with other AI tools

## Support

For issues or questions:

- GitHub Issues: https://github.com/Djeisen642/local-ai-python/issues
- Documentation: See README.md and other docs/
- Ollama Docs: https://ollama.com/docs
