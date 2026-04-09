# LearningMachina — Voice Architecture

The voice app gives the LearningMachina robot the ability to talk.
It captures speech from a microphone, converts it to text, sends the question to the [learning-hub](https://github.com/LearningMachina/learning-hub) API, and speaks the answer back through a speaker.

---

## High-Level Flow

```
                              voice app
┌─────┐    audio    ┌──────────────────────────────────┐   HTTP/JSON  ┌──────────────┐
│ mic │ ──────────▶ │  Whisper (STT)                   │              │              │
└─────┘             │         │                        │  /api/ask/   │ learning-hub │
                    │         ▼                        │───stream───▶ │   (Fastify)  │
                    │   question text                  │              │              │
                    │         │         answer text    │ ◀─────────── │              │
                    │         ▼              │         │              └──────────────┘
┌─────┐    audio    │  Piper / Kokoro (TTS)  │         │
│ spk │ ◀────────── │         ▲──────────────┘         │
└─────┘             └──────────────────────────────────┘
```

1. **Listen** — The microphone captures the student's speech.
2. **STT** — Whisper transcribes the audio into text.
3. **Ask** — The text is sent to the learning-hub streaming endpoint (`POST /api/ask/stream`).
4. **TTS** — As answer chunks stream back, Piper or Kokoro converts them to speech.
5. **Speak** — The synthesised audio is played through the speaker.

Because the learning-hub streams its response via Server-Sent Events, the robot can begin speaking before the full answer has been generated.

---

## Target Platforms

The voice app is designed to run on any Linux machine with a mic, speaker, and NVIDIA GPU.
Two deployment targets are planned; the notebook serves as the development and interim production platform.

### Current — Notebook (RTX 2060)

| Aspect | Detail |
|--------|--------|
| **GPU** | NVIDIA RTX 2060 — 6 GB dedicated VRAM, Turing, compute capability 7.5 |
| **OS** | Ubuntu (x86_64) with CUDA drivers installed |
| **Python** | 3.12 (venv) |
| **RAM** | System RAM + 6 GB VRAM — enough to run Whisper `medium` alongside Kokoro comfortably |

This is the recommended platform for now. The dedicated 6 GB VRAM removes the memory pressure of the Jetson Nano and allows larger, more accurate models.

### Future — Jetson Orin Nano

| Aspect | Detail |
|--------|--------|
| **GPU** | NVIDIA Ampere — up to 1024 CUDA cores, compute capability 8.7 |
| **OS** | Ubuntu 22.04 (aarch64, JetPack 6.x) |
| **Python** | 3.10+ |
| **RAM** | 4–8 GB shared — model sizes need care, but far more capable than the original Nano |

The Orin Nano is the planned robot-embedded target. JetPack 6.x ships Ubuntu 22.04 with modern CUDA support, making it a straightforward migration from the notebook.

### Why not the original Jetson Nano?

The original Jetson Nano (Maxwell, 128 CUDA cores, compute capability 5.3) has significant limitations:

- **OS:** JetPack 4.6 only supports Ubuntu 18.04 / Python 3.6. Running Ubuntu 22.04 requires an unsupported community image.
- **CUDA:** Compute capability 5.3 is being dropped by newer versions of PyTorch, CTranslate2, and ONNX Runtime. Building compatible wheels becomes increasingly difficult.
- **Memory:** 4 GB shared between CPU and GPU leaves very little room for Whisper + TTS + the OS.
- **Performance:** 128 CUDA cores make real-time STT + TTS marginal at best.

It can work for basic testing with `whisper tiny` + Piper, but is not recommended for production use.

---

## Speech-to-Text — Whisper

[Whisper](https://github.com/openai/whisper) is an open-source automatic speech recognition model by OpenAI.

| Aspect | Detail |
|--------|--------|
| **Model** | `whisper` (size chosen at deploy time — `base`, `small`, `medium`, …) |
| **Input** | Raw audio from the microphone (16 kHz mono WAV) |
| **Output** | Transcribed text |
| **Runtime** | Runs locally via `faster-whisper` (CTranslate2 backend, CUDA-accelerated on the Jetson) |

On the notebook, `faster-whisper` runs with CUDA out of the box (standard PyPI wheels support compute capability 7.5). The `base`, `small`, or even `medium` models are all viable with 6 GB VRAM.

### Wake-word / Voice Activity Detection

Before sending audio to Whisper the app uses a lightweight voice-activity detector (VAD) to decide when the student has started and stopped speaking.
This avoids transcribing silence and keeps latency low by only feeding relevant audio segments to the model.

---

## Text-to-Speech — Piper & Kokoro

The app supports two TTS engines, selectable via configuration.

### Piper

[Piper](https://github.com/rhasspy/piper) is a fast, local neural TTS system optimised for single-board computers like the Raspberry Pi.

| Aspect | Detail |
|--------|--------|
| **Voices** | ONNX voice models (`.onnx` + `.onnx.json`) |
| **Latency** | Very low — suitable for real-time chunk-by-chunk synthesis |
| **Input** | Plain text |
| **Output** | PCM audio (16-bit, configurable sample rate) |

### Kokoro

[Kokoro](https://github.com/hexgrad/kokoro) is a high-quality neural TTS engine that produces natural-sounding speech.

| Aspect | Detail |
|--------|--------|
| **Voices** | Pre-trained voice packs |
| **Quality** | Higher naturalness compared to Piper, at the cost of more compute |
| **Input** | Plain text |
| **Output** | PCM audio |

### Choosing an engine

| Criterion | Piper | Kokoro |
|-----------|-------|--------|
| Speed on Jetson Nano | ✅ Faster (ONNX, CPU-optimised) | Slower (PyTorch + CUDA) |
| Speed on notebook (RTX 2060) | ✅ Fast | ✅ Fast (CUDA) |
| Voice quality | Good | ✅ More natural |
| Resource usage | ✅ Lower (~100 MB) | Higher (PyTorch + model weights) |
| GPU required | No | Recommended (CUDA) |

On the **notebook** either engine works well — Kokoro is the better default since the RTX 2060 has plenty of VRAM.
On the **Orin Nano** (future), prefer Piper to leave GPU headroom for Whisper, or benchmark Kokoro once the hardware is available.

---

## Integration with Learning Hub

The voice app is a client of the learning-hub REST API. It uses two endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /api/ask/stream` | Primary — streams answer chunks so TTS can start immediately |
| `POST /api/ask` | Fallback — returns the complete answer in one response |

Conversation context is maintained by passing `conversation_id` back to the hub on follow-up questions, allowing multi-turn dialogues.

See the [learning-hub ARCHITECTURE.md](https://github.com/LearningMachina/learning-hub/blob/main/ARCHITECTURE.md) for the full API reference.

---

## Audio I/O

| Component | Detail |
|-----------|--------|
| **Capture** | System microphone via PortAudio / PyAudio (16 kHz, mono) |
| **Playback** | System speaker via PortAudio / PyAudio or SDL2 |
| **Format** | 16-bit PCM internally; no lossy compression on the local audio path |

Input and output devices are configured independently so they can be different physical devices.

### Recommended setup (notebook)

| Role | Device | Why |
|------|--------|-----|
| **Mic (input)** | Laptop built-in microphone | Consistent 16 kHz capture, no Bluetooth latency |
| **Speaker (output)** | JBL Go via Bluetooth (A2DP) | Better sound quality than laptop speakers for TTS playback |

### Bluetooth caveat

When a Bluetooth device is used for **both** input and output, PulseAudio / PipeWire switches from the A2DP profile (high-quality stereo output, no mic) to the HFP profile (mic enabled, but audio drops to **8 kHz mono**).
This degrades Whisper accuracy since it expects 16 kHz input.

To avoid this, keep capture on the laptop mic and only use the Bluetooth speaker for output. The app supports this via separate `AUDIO_INPUT_DEVICE` and `AUDIO_OUTPUT_DEVICE` settings.

---

## Component Overview

```
voice/
├── stt/              # Whisper integration and VAD
├── tts/              # Piper and Kokoro engine wrappers
├── hub_client/       # HTTP client for learning-hub API
├── audio/            # Microphone capture and speaker playback
├── config/           # Configuration loading (env / YAML)
└── main.py           # Entry point — orchestrates the listen → ask → speak loop
```

---

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `HUB_URL` | `http://localhost:3000` | Learning-hub base URL |
| `TTS_ENGINE` | `piper` | TTS backend: `piper` or `kokoro` |
| `PIPER_MODEL_PATH` | — | Path to the Piper ONNX voice model |
| `KOKORO_MODEL_PATH` | — | Path to the Kokoro voice pack |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `AUDIO_INPUT_DEVICE` | system default | Microphone device name or index |
| `AUDIO_OUTPUT_DEVICE` | system default | Speaker device name or index |
| `VAD_THRESHOLD` | `0.5` | Voice-activity detection sensitivity (0–1) |

---

## Tech Stack

- **Language:** Python
- **STT:** Whisper (via `faster-whisper`)
- **TTS:** Piper / Kokoro
- **Audio:** PyAudio (PortAudio bindings)
- **HTTP:** `httpx` (async, with SSE streaming support)
- **Target hardware:** Notebook with RTX 2060 (current), Jetson Orin Nano (future)

### Platform Notes

- **Notebook (RTX 2060):** Standard PyPI wheels work out of the box. Use a `venv` with Python 3.12. All dependencies (`faster-whisper`, `piper-tts`, `kokoro`, `httpx`, `PyAudio`) install cleanly via pip.
- **Jetson Orin Nano (future):** NVIDIA provides Jetson-specific PyTorch wheels — use those instead of stock PyPI packages. Piper needs `onnxruntime` built for aarch64 (pre-built wheels available from ONNX Runtime releases or the Jetson Zoo). With 4–8 GB shared RAM, plan model loading carefully.
