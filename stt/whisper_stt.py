"""Whisper-based speech-to-text and a console fallback for testing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


# -- abstract base -----------------------------------------------------------

class BaseSTT(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load / warm-up the model."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return transcribed text from a 16-bit int16 numpy array."""


# -- Whisper -----------------------------------------------------------------

class WhisperSTT(BaseSTT):
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device = self._cfg.whisper_device
        if device == "auto":
            device = "cuda"  # fall back handled by faster-whisper
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading Whisper model=%s device=%s compute=%s",
            self._cfg.whisper_model, device, compute_type,
        )
        self._model = WhisperModel(
            self._cfg.whisper_model,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        assert self._model is not None, "call load() first"
        # faster-whisper expects float32 normalised to [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_f32,
            language=self._cfg.whisper_language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info("Transcribed (%s, %.1fs): %s", info.language, info.duration, text)
        return text


# -- Console fallback --------------------------------------------------------

class ConsoleSTT(BaseSTT):
    """Reads questions from stdin — useful for testing without a microphone."""

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # audio argument is ignored in console mode
        try:
            return input("\n🎤 You: ").strip()
        except EOFError:
            return ""


# -- factory -----------------------------------------------------------------

def create_stt_engine(cfg: Config) -> BaseSTT:
    match cfg.stt_engine:
        case "whisper":
            return WhisperSTT(cfg)
        case "console":
            return ConsoleSTT()
        case other:
            raise ValueError(f"Unknown STT engine: {other!r}")
