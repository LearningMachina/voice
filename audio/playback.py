"""Speaker playback via PyAudio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pyaudio

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class Speaker:
    """Plays PCM audio through the system speaker."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pa = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None

    def open(self) -> None:
        kwargs: dict = dict(
            format=pyaudio.paInt16,
            channels=self._cfg.channels,
            rate=self._cfg.sample_rate,
            output=True,
            frames_per_buffer=1024,
        )
        if self._cfg.audio_output_device is not None:
            kwargs["output_device_index"] = self._cfg.audio_output_device
        self._stream = self._pa.open(**kwargs)
        logger.info(
            "Speaker opened (rate=%d, device=%s)",
            self._cfg.sample_rate,
            self._cfg.audio_output_device or "default",
        )

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        try:
            self._pa.terminate()
        except Exception:
            pass

    def play(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Play a numpy int16 audio array.

        If *sample_rate* differs from the configured rate the caller is
        responsible for resampling beforehand.
        """
        assert self._stream is not None, "call open() first"
        self._stream.write(audio.astype(np.int16).tobytes())

    def play_bytes(self, data: bytes) -> None:
        """Play raw PCM bytes directly."""
        assert self._stream is not None, "call open() first"
        self._stream.write(data)
