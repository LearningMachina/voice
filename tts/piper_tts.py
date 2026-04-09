"""Piper neural TTS engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseTTS, TTSResult

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class PiperTTS(BaseTTS):
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._voice = None
        self._sr = 22050  # default, overwritten on load

    def load(self) -> None:
        from piper import PiperVoice

        model_path = self._cfg.piper_model_path
        if not model_path:
            raise RuntimeError(
                "PIPER_MODEL_PATH must be set when TTS_ENGINE=piper. "
                "Download a voice from https://github.com/rhasspy/piper/blob/master/VOICES.md"
            )
        logger.info("Loading Piper voice from %s", model_path)
        self._voice = PiperVoice.load(model_path)
        self._sr = self._voice.config.sample_rate
        logger.info("Piper ready (sample_rate=%d)", self._sr)

    def synthesize(self, text: str) -> TTSResult:
        assert self._voice is not None, "call load() first"
        # synthesize_stream_raw yields bytes chunks of raw PCM (int16)
        chunks: list[bytes] = []
        for audio_bytes in self._voice.synthesize_stream_raw(text):
            chunks.append(audio_bytes)
        raw = b"".join(chunks)
        audio = np.frombuffer(raw, dtype=np.int16)
        return TTSResult(audio=audio, sample_rate=self._sr)

    @property
    def sample_rate(self) -> int:
        return self._sr
