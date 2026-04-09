"""Kokoro neural TTS engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseTTS, TTSResult

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000


class KokoroTTS(BaseTTS):
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pipeline = None
        self._sr = KOKORO_SAMPLE_RATE

    def load(self) -> None:
        from kokoro import KPipeline

        logger.info(
            "Loading Kokoro (lang=%s, voice=%s)",
            self._cfg.kokoro_lang_code,
            self._cfg.kokoro_voice,
        )
        self._pipeline = KPipeline(
            lang_code=self._cfg.kokoro_lang_code,
            repo_id="hexgrad/Kokoro-82M",
        )
        # Pre-load the voice so it's cached for subsequent calls
        self._pipeline.load_voice(self._cfg.kokoro_voice)
        logger.info("Kokoro ready (sample_rate=%d)", self._sr)

    def synthesize(self, text: str) -> TTSResult:
        assert self._pipeline is not None, "call load() first"
        all_samples: list[np.ndarray] = []

        try:
            for result in self._pipeline(
                text,
                voice=self._cfg.kokoro_voice,
                speed=self._cfg.kokoro_speed,
            ):
                # KPipeline yields Result objects with .audio property
                # (.audio is a torch.FloatTensor or None)
                audio = result.audio if hasattr(result, "audio") else result[2]

                if audio is None:
                    logger.debug("Kokoro chunk had no audio (phonemes=%s)", result.phonemes[:40] if hasattr(result, "phonemes") else "?")
                    continue

                # Convert torch tensor → numpy float32
                if hasattr(audio, "detach"):
                    arr = audio.detach().cpu().numpy()
                elif hasattr(audio, "numpy"):
                    arr = audio.numpy()
                elif isinstance(audio, np.ndarray):
                    arr = audio
                else:
                    logger.warning("Kokoro yielded unexpected audio type: %s", type(audio).__name__)
                    continue

                arr = np.atleast_1d(arr).flatten().astype(np.float32)
                if arr.size > 0:
                    all_samples.append(arr)
        except Exception:
            logger.exception("Kokoro synthesis failed for: %r", text[:80])

        if not all_samples:
            logger.warning("Kokoro produced no audio for: %r", text[:80])
            return TTSResult(audio=np.array([], dtype=np.int16), sample_rate=self._sr)

        # Kokoro returns float32 in [-1, 1]; convert to int16
        combined = np.concatenate(all_samples)
        logger.debug("Kokoro synthesised %d samples (%.2fs)", len(combined), len(combined) / self._sr)
        audio_int16 = (combined * 32767).clip(-32768, 32767).astype(np.int16)
        return TTSResult(audio=audio_int16, sample_rate=self._sr)

    @property
    def sample_rate(self) -> int:
        return self._sr
