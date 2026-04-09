# LearningMachina — Voice

Voice interface for the LearningMachina platform.
Captures speech from a microphone, transcribes it with Whisper, sends the question to the [learning-hub](https://github.com/LearningMachina/learning-hub) API, and speaks the answer back using Piper or Kokoro TTS.

## Prerequisites

- **Python 3.10+** (tested with 3.12)
- **PortAudio** system library (required by PyAudio)
- **NVIDIA GPU + CUDA drivers** (recommended for Whisper and Kokoro)

### Install PortAudio

```bash
# Ubuntu / Debian
sudo apt-get install portaudio19-dev

# macOS
brew install portaudio
```

## Setup

```bash
# Clone and enter the project
cd voice

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

> **Note:** `piper-tts` and `kokoro` are included in requirements.txt but are
> only needed if you plan to use that TTS engine. You can comment out the one
> you don't need to speed up installation.

### Configuration

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUB_URL` | `http://localhost:3000` | Learning-hub API URL |
| `STT_ENGINE` | `whisper` | `whisper` or `console` (type questions via keyboard) |
| `TTS_ENGINE` | `console` | `piper`, `kokoro`, or `console` (print answers to terminal) |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium` |

See [.env.example](.env.example) for the full list.

### List audio devices

To find device indices for `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE`:

```bash
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    direction = 'IN' if info['maxInputChannels'] > 0 else 'OUT'
    print(f'{i:2d} [{direction}] {info[\"name\"]}')
p.terminate()
"
```

## Running

### Quick start (console mode — no mic/speaker/models needed)

The easiest way to test the pipeline. Questions are typed, answers are printed:

```bash
# Terminal 1 — start the fake learning-hub API
python -m fake_hub.server

# Terminal 2 — run the voice app in full console mode
STT_ENGINE=console TTS_ENGINE=console python main.py
```

### With Whisper STT + console TTS

Speak into the microphone; answers are printed (no TTS models needed):

```bash
# Terminal 1
python -m fake_hub.server

# Terminal 2
STT_ENGINE=whisper TTS_ENGINE=console WHISPER_MODEL=base python main.py
```

### Full voice mode (Whisper + Piper)

```bash
# Download a Piper voice model first
# See: https://github.com/rhasspy/piper/blob/master/VOICES.md
# Example: en_US-lessac-medium

STT_ENGINE=whisper TTS_ENGINE=piper \
  PIPER_MODEL_PATH=/path/to/en_US-lessac-medium.onnx \
  python main.py
```

### Full voice mode (Whisper + Kokoro)

```bash
STT_ENGINE=whisper TTS_ENGINE=kokoro \
  KOKORO_VOICE=af_heart \
  python main.py
```

### Switching to the real learning-hub

Once the learning-hub is deployed, just change the URL:

```bash
HUB_URL=http://<learning-hub-host>:3000 python main.py
```

No code changes needed — the fake hub and real hub expose the same API.

## Fake Hub Server

A lightweight stand-in for the real learning-hub, returning canned answers.
Implements the endpoints the voice app uses:

- `GET  /api/health`
- `POST /api/ask`
- `POST /api/ask/stream` (SSE)

```bash
python -m fake_hub.server              # default port 3000
python -m fake_hub.server --port 8080  # custom port
```

## Project Structure

```
voice/
├── main.py              # Entry point — conversation loop
├── config.py            # Environment-based configuration
├── audio/
│   ├── capture.py       # Microphone recording with VAD
│   └── playback.py      # Speaker playback
├── stt/
│   └── whisper_stt.py   # Whisper STT + console fallback
├── tts/
│   ├── base.py          # Abstract TTS interface
│   ├── piper_tts.py     # Piper TTS engine
│   ├── kokoro_tts.py    # Kokoro TTS engine
│   └── console_tts.py   # Console fallback (print to stdout)
├── hub_client/
│   └── client.py        # Async HTTP client with SSE parsing
├── fake_hub/
│   └── server.py        # Fake learning-hub for development
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md
└── LICENSE
```

## Further Reading

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design, platform notes, and component details.

## License

[MIT](LICENSE)
