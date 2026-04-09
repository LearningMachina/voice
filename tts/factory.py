"""TTS engine factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseTTS
from .console_tts import ConsoleTTS
from .kokoro_tts import KokoroTTS
from .piper_tts import PiperTTS

if TYPE_CHECKING:
    from config import Config


def create_tts_engine(cfg: Config) -> BaseTTS:
    match cfg.tts_engine:
        case "piper":
            return PiperTTS(cfg)
        case "kokoro":
            return KokoroTTS(cfg)
        case "console":
            return ConsoleTTS()
        case other:
            raise ValueError(f"Unknown TTS engine: {other!r}")
