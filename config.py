"""Voice app configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class Config:
    # Learning-hub connection
    hub_url: str = field(default_factory=lambda: _env("HUB_URL", "http://localhost:3000"))

    # STT
    stt_engine: str = field(default_factory=lambda: _env("STT_ENGINE", "whisper"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "auto"))
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "en"))

    # TTS
    tts_engine: str = field(default_factory=lambda: _env("TTS_ENGINE", "console"))
    piper_model_path: str = field(default_factory=lambda: _env("PIPER_MODEL_PATH", ""))
    kokoro_voice: str = field(default_factory=lambda: _env("KOKORO_VOICE", "af_heart"))
    kokoro_lang_code: str = field(default_factory=lambda: _env("KOKORO_LANG_CODE", "a"))
    kokoro_speed: float = field(default_factory=lambda: _env_float("KOKORO_SPEED", 1.0))

    # Audio
    audio_input_device: int | None = field(default_factory=lambda: (
        int(_env("AUDIO_INPUT_DEVICE")) if _env("AUDIO_INPUT_DEVICE") else None
    ))
    audio_output_device: int | None = field(default_factory=lambda: (
        int(_env("AUDIO_OUTPUT_DEVICE")) if _env("AUDIO_OUTPUT_DEVICE") else None
    ))
    sample_rate: int = field(default_factory=lambda: _env_int("SAMPLE_RATE", 16000))
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM

    # VAD
    vad_threshold: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD", 0.02))
    vad_silence_duration: float = field(default_factory=lambda: _env_float("VAD_SILENCE_DURATION", 1.5))
    vad_min_speech_duration: float = field(default_factory=lambda: _env_float("VAD_MIN_SPEECH_DURATION", 0.3))

    # General
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


def load_config(env_file: str | Path | None = ".env") -> Config:
    """Load configuration from environment (optionally reading a .env file first)."""
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)
    return Config()
