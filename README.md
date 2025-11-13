# Local AI

A privacy-focused AI application that provides AI capabilities without requiring internet connectivity. Features real-time speech-to-text using Whisper and intelligent task management with local LLM.

## Table of Contents

- [Core Features](#core-features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Audio Debugging](#audio-debugging)
- [Performance](#performance)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Documentation

- **[Task Management Guide](docs/task-management-guide.md)** - Complete guide to voice task detection and MCP integration
- **[Task Management Deployment](docs/task-management-deployment.md)** - Deployment instructions for Ollama and MCP server
- **[Task Management Configuration](docs/task-management-configuration.md)** - Detailed configuration options and integration patterns

## Core Features

### 🎤 Real-time Speech Transcription

Convert voice to text with local processing using OpenAI's Whisper model:

- **Real-time transcription** - See your words appear as you speak
- **Voice activity detection** - Automatically detects when speech is present
- **GPU acceleration** - Automatic GPU detection with CPU fallback
- **Privacy-first** - All processing happens locally, no data leaves the machine
- **Linux optimized** - Designed for Linux environments with headless support

### ✅ Voice Task Management

Automatically detect and manage tasks from voice input using local LLM:

- **Automatic task detection** - AI identifies tasks from natural speech
- **Local LLM processing** - Uses Ollama with llama3.2:3b for privacy
- **Persistent storage** - SQLite database with full history tracking
- **MCP integration** - Expose tasks to AI tools via Model Context Protocol
- **Source-agnostic** - Works with voice, CLI, or direct API calls

### 🔮 Planned Features

- Text-to-speech synthesis
- Email notification intelligence
- Calendar event prioritization
- Adaptive learning and personalization
- Smart home device integration

## Architecture Philosophy

The system is built with extensibility in mind, using abstract interfaces and plugin-style architecture to enable future AI assistant workflows including embedding generation, response generation, and text-to-speech capabilities.

## Quick Start

### Prerequisites

- **Python 3.13+** (strict requirement)
- **uv** (modern Python package manager)
- Microphone access
- (Optional) NVIDIA GPU with CUDA for faster processing
- (Optional) **Ollama** for task management features

**Install uv package manager:**

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

**Install Ollama (for task management):**

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required model (2GB download)
ollama pull llama3.2:3b
```

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Djeisen642/local-ai-python.git
   cd local-ai-python
   ```

2. **Install system dependencies (Linux):**

   ```bash
   sudo apt update
   sudo apt install python3-dev portaudio19-dev
   ```

3. **Install the package:**

   ```bash
   uv pip install -e .
   ```

4. **For GPU acceleration (optional):**
   - Install CUDA Toolkit 11.8 or 12.x from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
   - The application will automatically detect and use GPU if available

### Usage

#### Command Line Interface

Start real-time speech-to-text:

```bash
# After installation, you can use either:
local-ai

# Or the module approach:
python -m local_ai.main
```

**Controls:**

- Speak into your microphone to see real-time transcription
- Press `Ctrl+C` to stop

**Example output:**

```
🎤 Starting speech-to-text service...
✅ Listening for speech. Speak into your microphone!
   Press Ctrl+C to stop.
📝 [1] Hello, this is a test of the speech recognition system. (87%)
📝 [2] It works pretty well with local AI models. (92%)
```

**CLI Options:**

```bash
# Hide confidence percentages
local-ai --no-confidence

# Enable verbose logging
local-ai --verbose

# Force CPU-only mode (disable GPU)
local-ai --force-cpu

# Enable audio debugging (save processed audio to WAV files)
local-ai --debug-audio

# Enable audio debugging with custom output directory
local-ai --debug-audio --debug-audio-dir /path/to/output
```

#### Python API

**Speech-to-Text:**

```python
import asyncio
from local_ai.speech_to_text.service import SpeechToTextService
from local_ai.speech_to_text.models import TranscriptionResult

async def main():
    service = SpeechToTextService()

    # Set up callback for transcription results with confidence
    def on_transcription_result(result: TranscriptionResult):
        print(f"Transcribed: {result.text} (confidence: {result.confidence:.1%})")

    service.set_transcription_result_callback(on_transcription_result)

    # Start listening
    await service.start_listening()

    # Keep running
    await asyncio.sleep(10)  # Listen for 10 seconds

    # Stop
    await service.stop_listening()

asyncio.run(main())
```

**Task Management:**

See [Task Management Guide](docs/task-management-guide.md) for complete API documentation.

## Task Management

Automatically detect and manage tasks from voice input using local LLM. See the **[Task Management Guide](docs/task-management-guide.md)** for complete documentation.

### Quick Start

```bash
# Install Ollama and model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Start application
local-ai

# Say: "Remind me to finish the report by Friday"
# Task is automatically detected and stored
```

### Key Features

- **Automatic Detection** from speech or text
- **Local LLM** processing with Ollama (llama3.2:3b)
- **MCP Integration** for Claude Desktop, Cursor, etc.
- **Smart Extraction** of description, priority, and due dates

See [Task Management Guide](docs/task-management-guide.md) for API examples, configuration, and troubleshooting.

## Configuration

The speech-to-text system uses optimized defaults but can be customized by modifying the configuration constants in `src/local_ai/speech_to_text/config.py`:

### Model Optimization

**English-Only Models** (enabled by default):

```python
# In src/local_ai/speech_to_text/config.py
OPT_USE_ENGLISH_ONLY_MODEL = True  # Use .en models for better performance
```

**Distilled Models** (experimental, disabled by default):

```python
# In src/local_ai/speech_to_text/config.py
OPT_USE_DISTILLED_MODELS = True  # Enable faster distilled models

# When enabled:
# - tiny/small/medium → distil-medium.en (faster than small.en)
# - large → distil-large-v3 (much faster than large-v3)
```

### Audio Settings

- **Sample Rate**: 16kHz (optimal for speech)
- **Chunk Size**: 1024 samples
- **Audio Format**: 16-bit PCM

### Model Selection

The system automatically selects optimal Whisper models based on your hardware:

- **English-Only Models** (default): Faster and more accurate for English speech
  - Uses `.en` suffix (e.g., `small.en`, `medium.en`)
  - ~2x faster than multilingual models
  - Automatically enabled for English-only use
- **Distilled Models** (experimental): Even faster with similar accuracy
  - `distil-medium.en`: ~2x faster than `medium.en`
  - `distil-large-v3`: ~6x faster than `large-v3`
  - Disabled by default, can be enabled in config

**Default Models:**

- **CPU**: `small.en` (244MB) - Best balance for most systems
- **4GB GPU**: `small.en` (244MB) - Fast with good accuracy
- **8GB+ GPU**: `medium.en` (769MB) - Better accuracy
- **16GB+ GPU**: `large` (1550MB) - Best accuracy

**Alternative Models:**

- `tiny.en` (39MB) - Fastest, lower accuracy
- `small.en` (244MB) - Recommended default
- `medium.en` (769MB) - Better accuracy
- `large` (1550MB) - Best accuracy (no .en variant)

### Voice Activity Detection

- **Frame Duration**: 30ms
- **Aggressiveness**: Level 2 (0-3 scale)
- **Min Speech Duration**: 0.5 seconds
- **Max Silence Duration**: 2.0 seconds

### Confidence Scoring

- **Range**: 0.0 to 1.0 (0% to 100%)
- **Calculation**: Based on Whisper's average log probability scores
- **Display**: Shown as percentage in CLI output (can be hidden with `--no-confidence`)
- **Typical Values**: 70-95% for clear speech, lower for noisy or unclear audio

## Audio Debugging

The audio debugging feature allows you to capture and save the processed audio that is sent to the Whisper transcription model. This is useful for diagnosing transcription issues, understanding audio preprocessing effects, and validating audio quality.

### Enabling Audio Debugging

**Via Command Line:**

```bash
# Enable audio debugging with default directory
local-ai --debug-audio

# Enable with custom output directory
local-ai --debug-audio --debug-audio-dir /tmp/audio_debug

# Combine with other options
local-ai --debug-audio --verbose --no-confidence
```

**Via Python API:**

```python
from pathlib import Path
from local_ai.speech_to_text.audio_debugger import AudioDebugger
from local_ai.speech_to_text.transcriber import WhisperTranscriber

# Create audio debugger
debugger = AudioDebugger(
    enabled=True,
    output_dir=Path("/tmp/audio_debug")
)

# Create transcriber with debugger
transcriber = WhisperTranscriber(
    model_size="small",
    audio_debugger=debugger
)

# Use normally - audio will be saved automatically
result = await transcriber.transcribe_audio_with_result(audio_data)
```

### Output Files

**Default Location:**

```
~/.cache/local_ai/audio_debug/
```

**File Naming Convention:**

```
audio_{YYYYMMDD}_{HHMMSS}_{duration_ms}.wav
```

**Example:**

```
audio_20241112_143022_500ms.wav
```

- **Date**: November 12, 2024
- **Time**: 14:30:22 (24-hour format)
- **Duration**: 500 milliseconds

**WAV File Format:**

- **Sample Rate**: 16000 Hz (matches Whisper input)
- **Channels**: 1 (mono)
- **Sample Width**: 16-bit PCM
- **Format**: Standard WAV with proper headers

These files can be played back with any audio player or analyzed with audio tools like Audacity.

### Usage Examples

**Debugging Transcription Accuracy:**

```bash
# Enable audio debugging
local-ai --debug-audio

# Speak into microphone
# Check saved WAV files to verify audio quality
ls -lh ~/.cache/local_ai/audio_debug/

# Play back a saved file
aplay ~/.cache/local_ai/audio_debug/audio_20241112_143022_500ms.wav
```

**Analyzing Audio Preprocessing:**

```bash
# Save to custom directory for analysis
local-ai --debug-audio --debug-audio-dir ./audio_analysis

# Use audio analysis tools
ffprobe ./audio_analysis/audio_20241112_143022_500ms.wav
```

**Comparing Different Configurations:**

```bash
# Test with different settings
local-ai --debug-audio --debug-audio-dir ./test1
# Change configuration, then:
local-ai --debug-audio --debug-audio-dir ./test2
# Compare audio files between directories
```

### Troubleshooting Audio Debugging

#### Directory Permission Issues

**Error: "Permission denied" when creating debug directory**

```bash
# Check directory permissions
ls -ld ~/.cache/local_ai/

# Fix permissions
chmod 755 ~/.cache/local_ai/

# Or use a different directory
local-ai --debug-audio --debug-audio-dir /tmp/audio_debug
```

#### Disk Space Issues

**Error: "No space left on device"**

```bash
# Check available disk space
df -h ~/.cache/local_ai/

# Clean up old debug files
rm ~/.cache/local_ai/audio_debug/*.wav

# Or use a directory on a different partition
local-ai --debug-audio --debug-audio-dir /mnt/storage/audio_debug
```

#### Files Not Being Created

**Audio debugging enabled but no files appear:**

1. **Verify debugging is enabled:**

   ```bash
   # Check for debug messages in verbose mode
   local-ai --debug-audio --verbose
   ```

2. **Check directory exists and is writable:**

   ```bash
   # Verify directory
   ls -ld ~/.cache/local_ai/audio_debug/

   # Test write permissions
   touch ~/.cache/local_ai/audio_debug/test.txt
   rm ~/.cache/local_ai/audio_debug/test.txt
   ```

3. **Verify speech is being detected:**
   - Ensure microphone is working
   - Check that VAD is detecting speech
   - Look for transcription output in console

#### Audio Quality Issues in Saved Files

**Saved audio sounds distorted or incorrect:**

- Files are saved in the exact format sent to Whisper (16kHz, mono, 16-bit)
- If audio sounds wrong, this indicates a preprocessing issue
- Check microphone input levels and quality
- Review VAD settings and audio filtering configuration

**Files are silent or very quiet:**

- Check microphone input volume in system settings
- Verify audio normalization is working correctly
- Test with `arecord` to ensure microphone is capturing audio

### Performance Impact

**When Disabled (Default):**

- Zero performance overhead
- Single boolean check per transcription

**When Enabled:**

- Minimal impact: typically <10ms per transcription
- Synchronous file writes (fast for small WAV files)
- Errors don't interrupt transcription
- No impact on transcription accuracy or latency

### Privacy and Security

**Privacy Considerations:**

- Audio files contain raw speech data
- Files are stored locally (not transmitted)
- Default directory is user-specific (`~/.cache/local_ai/`)
- Consider data sensitivity when enabling

**File Permissions:**

- Files are created with user-only read/write permissions (0600)
- Directory is created with standard permissions (0755)
- No automatic cleanup - user manages files manually

**Disk Space Management:**

- Files are typically small (50-200KB each)
- No automatic cleanup or rotation
- User should periodically clean up debug directory
- Consider disk space when enabling for extended periods

**Cleanup Commands:**

```bash
# Remove all debug files
rm -rf ~/.cache/local_ai/audio_debug/

# Remove files older than 7 days
find ~/.cache/local_ai/audio_debug/ -name "*.wav" -mtime +7 -delete

# Remove files older than 1 day
find ~/.cache/local_ai/audio_debug/ -name "*.wav" -mtime +1 -delete

# Check total size
du -sh ~/.cache/local_ai/audio_debug/
```

## Performance

### Hardware Requirements

**Minimum:**

- 4GB RAM
- Any modern CPU
- Microphone

**Recommended:**

- 8GB RAM
- NVIDIA GPU with 4GB+ VRAM
- Quality microphone

### Expected Performance

| Hardware  | Model     | Latency | Accuracy | Notes                       |
| --------- | --------- | ------- | -------- | --------------------------- |
| CPU Only  | small.en  | 2-3s    | Good     | English-only, optimized     |
| 4GB GPU   | small.en  | 1-1.5s  | Good     | English-only, fast          |
| 8GB GPU   | medium.en | 1.5-2s  | Better   | English-only, accurate      |
| 16GB+ GPU | large     | 2-3s    | Best     | Multilingual, most accurate |

**With Distilled Models (experimental):**
| Hardware | Model | Latency | Accuracy | Speedup |
| --------- | ------------------ | ------- | -------- | ------- |
| CPU Only | distil-medium.en | 1.5-2s | Good | ~2x |
| 8GB GPU | distil-medium.en | 0.8-1s | Better | ~2x |
| 16GB+ GPU | distil-large-v3 | 1-1.5s | Best | ~6x |

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
uv pip install -e .[dev]
```

### Running Tests

The project uses pytest with parallel execution support for faster testing:

```bash
# Run all tests in parallel
pytest -n 2

# Run with coverage
pytest -n 2 --cov=src --cov-report=html --cov-fail-under=90

# Run specific test categories
pytest -n 2 -m unit            # Fast unit tests only
pytest -n 2 -m integration     # Integration tests only
pytest -n 2 -m performance     # Performance benchmarks

# Recommended workflow: run unit tests first, then integration
pytest -n 2 -m unit && pytest -n 2 -m integration

# TDD workflow (stop on first failure)
pytest -n 2 -x --cov=src
```

### Code Quality

```bash
# Run type checking
mypy src tests

# Run linting and formatting
ruff check src tests
ruff format src tests

# Run security checks
bandit -r src

# Run all quality checks
mypy src tests && ruff check src tests && ruff format src tests && bandit -r src
```

## Troubleshooting

### Common Issues

#### 🎤 Microphone Problems

**"No microphone detected" or "Permission denied"**

_Linux:_

```bash
# Check if microphone is detected
arecord -l

# Test microphone recording
arecord -d 5 test.wav && aplay test.wav

# Fix permissions (add user to audio group)
sudo usermod -a -G audio $USER
# Log out and back in for changes to take effect
```

**"Device busy" or "Audio device in use"**

- Close other applications using the microphone (Zoom, Discord, etc.)
- Restart the audio service:

  ```bash
  # Linux
  sudo systemctl restart pulseaudio
  ```

#### 🤖 Model and Performance Issues

**"Model not available" or slow first startup**

- The Whisper model downloads automatically on first use (~244MB for Small model)
- Ensure stable internet connection for initial download
- Models are cached in `~/.cache/huggingface/`

**GPU not detected or "CUDA out of memory"**

- Verify CUDA installation: `nvidia-smi`
- Check GPU memory: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`
- The application automatically falls back to CPU if GPU is unavailable
- For 4GB GPUs, consider using Whisper Tiny model instead of Small

**High CPU usage or slow transcription**

- CPU-only processing is normal but slower (3-5 seconds vs 1-2 seconds with GPU)
- Close unnecessary applications to free up CPU resources
- Consider upgrading to a GPU-enabled setup for better performance

#### 🔊 Audio Quality Issues

**Poor transcription accuracy**

- Ensure microphone is close to your mouth (6-12 inches)
- Reduce background noise
- Speak clearly and at normal pace
- Check microphone levels in system settings
- Try adjusting VAD sensitivity in configuration

**Transcription cuts off words or doesn't detect speech**

- Microphone levels may be too low - increase input volume
- Background noise may be interfering - try a quieter environment
- VAD sensitivity may be too high - check configuration settings

#### 📦 Installation Issues

**PyAudio installation fails**

_Linux:_

```bash
# Install development headers
sudo apt install python3-dev portaudio19-dev

# Alternative: use conda
conda install pyaudio
```

**faster-whisper installation issues**

- Ensure you have a compatible Python version (3.8-3.11)
- For GPU support, install CUDA Toolkit first
- Try installing without GPU support: `pip install faster-whisper --no-deps`

### Platform-Specific Notes

#### Linux

- **Headless servers**: Audio warnings are normal and don't affect functionality
- **PulseAudio**: Recommended audio system for best compatibility
- **Permissions**: User must be in `audio` group for microphone access
- **ALSA warnings**: Can be safely ignored in most cases

### Performance Optimization

#### GPU vs CPU Performance

**Standard English-Only Models:**

| Setup     | Whisper Model | Typical Latency | Memory Usage | Notes           |
| --------- | ------------- | --------------- | ------------ | --------------- |
| CPU Only  | tiny.en       | 1.5-2s          | 1GB RAM      | Fastest CPU     |
| CPU Only  | small.en      | 2-3s            | 2GB RAM      | Recommended CPU |
| 4GB GPU   | tiny.en       | 0.5s            | 500MB VRAM   | Very fast       |
| 4GB GPU   | small.en      | 1s              | 1GB VRAM     | Recommended     |
| 8GB GPU   | small.en      | 0.8s            | 1GB VRAM     | Fast & accurate |
| 8GB GPU   | medium.en     | 1.5s            | 2GB VRAM     | More accurate   |
| 16GB+ GPU | large         | 2-3s            | 4GB VRAM     | Best accuracy   |

**Distilled Models (Experimental - Enable in Config):**

| Setup     | Whisper Model    | Typical Latency | Memory Usage | Speedup |
| --------- | ---------------- | --------------- | ------------ | ------- |
| CPU Only  | distil-medium.en | 1.5-2s          | 2GB RAM      | ~2x     |
| 8GB GPU   | distil-medium.en | 0.8-1s          | 2GB VRAM     | ~2x     |
| 16GB+ GPU | distil-large-v3  | 1-1.5s          | 4GB VRAM     | ~6x     |

#### Model Selection Guide

**English-Only Models (Recommended for English):**

All models with `.en` suffix are optimized for English and provide ~2x speedup:

**Choose `tiny.en` if:**

- Limited GPU memory (2-4GB)
- Speed is critical
- Testing or development use

**Choose `small.en` if:** (Recommended Default)

- 4GB+ GPU memory available
- Best balance of speed and accuracy
- General purpose English transcription

**Choose `medium.en` if:**

- 8GB+ GPU memory available
- Higher accuracy needed
- Can accept slightly slower processing

**Choose `large` if:**

- 16GB+ GPU memory available
- Maximum accuracy required
- Need multilingual support (no .en variant)

**Distilled Models (Experimental):**

Enable in config for even faster processing:

- `distil-medium.en`: Best for most users (~2x faster than medium.en)
- `distil-large-v3`: For high-end GPUs (~6x faster than large-v3)

### Known Warnings (Safe to Ignore)

#### pkg_resources Deprecation Warning

```
UserWarning: pkg_resources is deprecated as an API...
```

This warning comes from the `webrtcvad` library dependency and does not affect functionality.

#### Audio System Warnings (Linux/Headless)

```
ALSA lib pcm.c:2721:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
Cannot connect to server socket err = No such file or directory
jack server is not running or cannot be started
```

These warnings appear when running in headless environments or systems without proper audio configuration. They don't prevent the application from working with available audio devices.

#### Test Runtime Warnings

```
RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
```

Minor test framework warnings that don't affect test results or application functionality.

#### 🤖 Task Management Issues

**"Ollama connection failed" or "Model not found"**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Pull the required model
ollama pull llama3.2:3b

# Verify model is available
ollama list
```

**Tasks not being detected from speech**

- Check that `TASK_DETECTION_FROM_SPEECH_ENABLED = True` in config
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check confidence threshold (default 0.7) - lower if needed
- Review logs with `--verbose` flag to see classification results

**MCP server not connecting**

```bash
# Check if port 3000 is available
netstat -tuln | grep 3000

# Verify MCP server configuration
python -m local_ai.task_management.mcp_server

# Check MCP client configuration (mcp.json)
```

**Database errors or corruption**

```bash
# Backup existing database
cp ~/.local-ai/tasks.db ~/.local-ai/tasks.db.backup

# Reset database (will lose tasks)
rm ~/.local-ai/tasks.db

# Database will be recreated on next run
```

**Low task detection accuracy**

- Ensure you're using llama3.2:3b (not smaller models)
- Speak clearly and use explicit task language ("I need to...", "Remind me to...")
- Check classification confidence in logs (`--verbose`)
- Consider adjusting `DEFAULT_CONFIDENCE_THRESHOLD` in config

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Run with verbose output to see detailed error messages
2. **Verify system requirements**: Ensure all dependencies are properly installed
3. **Test components individually**: Use the test suite to isolate problems
4. **Check hardware compatibility**: Verify microphone and GPU support
5. **Report bugs**: Create an issue on GitHub with system details and error logs

## Notes

- Based on [Python Boilerplate](https://github.com/smarlhens/python-boilerplate)
- Built with assistance from [Kiro](https://kiro.ai) - an AI-powered development assistant
