# Local AI

A privacy-focused AI application that provides AI capabilities without requiring internet connectivity. The primary focus is on speech-to-text functionality using OpenAI's Whisper model running locally.

## Core Features

### 🎤 Real-time Speech Transcription

Convert voice to text with local processing using OpenAI's Whisper model:

- **Real-time transcription** - See your words appear as you speak
- **Voice activity detection** - Automatically detects when speech is present
- **GPU acceleration** - Automatic GPU detection with CPU fallback
- **Privacy-first** - All processing happens locally, no data leaves the machine
- **Linux optimized** - Designed for Linux environments with headless support

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

**Install uv package manager:**

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
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
```

#### Python API

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

## Configuration

The speech-to-text system uses optimized defaults but can be customized by modifying the configuration constants in `src/local_ai/speech_to_text/config.py`:

### Audio Settings

- **Sample Rate**: 16kHz (optimal for speech)
- **Chunk Size**: 1024 samples
- **Audio Format**: 16-bit PCM

### Model Selection

- **Default**: Whisper Small (244MB) - Best balance for 8GB GPU
- **Alternatives**: Tiny (39MB), Medium (769MB), Large (1550MB)

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

| Hardware  | Model  | Latency | Accuracy |
| --------- | ------ | ------- | -------- |
| CPU Only  | Small  | 3-5s    | Good     |
| 8GB GPU   | Small  | 1-2s    | Good     |
| 8GB GPU   | Medium | 2-3s    | Better   |
| 16GB+ GPU | Large  | 2-4s    | Best     |

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

| Setup     | Whisper Model | Typical Latency | Memory Usage |
| --------- | ------------- | --------------- | ------------ |
| CPU Only  | Tiny          | 2-3s            | 1GB RAM      |
| CPU Only  | Small         | 4-6s            | 2GB RAM      |
| 4GB GPU   | Tiny          | 1s              | 500MB VRAM   |
| 4GB GPU   | Small         | 1.5s            | 1GB VRAM     |
| 8GB GPU   | Small         | 1s              | 1GB VRAM     |
| 8GB GPU   | Medium        | 1.5s            | 2GB VRAM     |
| 16GB+ GPU | Large         | 2s              | 4GB VRAM     |

#### Model Selection Guide

**Choose Whisper Tiny if:**

- Limited GPU memory (2-4GB)
- Speed is more important than accuracy
- Testing or development use

**Choose Whisper Small if:** (Recommended)

- 4GB+ GPU memory available
- Good balance of speed and accuracy needed
- General purpose use

**Choose Whisper Medium if:**

- 8GB+ GPU memory available
- Higher accuracy needed
- Can accept slightly slower processing

**Choose Whisper Large if:**

- 16GB+ GPU memory available
- Maximum accuracy required
- Processing time is not critical

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
