"""Microphone capture with energy-based voice activity detection."""

from __future__ import annotations

import logging
import struct
import math
import threading
from typing import TYPE_CHECKING

import numpy as np
import pyaudio

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)

CHUNK_FRAMES = 1024  # frames per read (~64 ms at 16 kHz)


class MicCapture:
    """Records speech from the microphone, using energy-based VAD to detect
    when the user starts and stops speaking."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pa = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None
        self._done = threading.Event()  # set when record_speech exits

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        kwargs: dict = dict(
            format=pyaudio.paInt16,
            channels=self._cfg.channels,
            rate=self._cfg.sample_rate,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )
        if self._cfg.audio_input_device is not None:
            kwargs["input_device_index"] = self._cfg.audio_input_device
        self._stream = self._pa.open(**kwargs)
        logger.info(
            "Mic opened (rate=%d, device=%s)",
            self._cfg.sample_rate,
            self._cfg.audio_input_device or "default",
        )

    def close(self) -> None:
        # 1. Signal record_speech to stop
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()  # unblocks pending read() in the other thread
            except Exception:
                pass
            # 2. Wait for the read thread to fully exit the C code
            self._done.wait(timeout=2.0)
            # 3. Now safe to free resources
            try:
                stream.close()
            except Exception:
                pass
        try:
            self._pa.terminate()
        except Exception:
            pass

    # -- VAD + recording -------------------------------------------------------

    def record_speech(self) -> np.ndarray:
        """Block until the user speaks, then return the recorded audio as a
        16-bit int16 numpy array (mono, at ``self._cfg.sample_rate``).

        Returns an empty array on interrupt so the caller can exit cleanly.
        """
        assert self._stream is not None, "call open() first"
        self._done.clear()

        threshold = self._cfg.vad_threshold
        silence_limit = self._cfg.vad_silence_duration
        min_speech = self._cfg.vad_min_speech_duration
        sr = self._cfg.sample_rate

        frames: list[bytes] = []
        speaking = False
        silent_chunks = 0
        chunks_per_sec = sr / CHUNK_FRAMES
        silence_chunks_limit = int(silence_limit * chunks_per_sec)
        min_speech_chunks = int(min_speech * chunks_per_sec)
        speech_chunks = 0

        logger.debug("Listening …")

        try:
            while self._stream is not None:
                try:
                    data = self._stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                except Exception:
                    break  # stream was aborted/closed

                rms = self._rms(data)

                if not speaking:
                    if rms >= threshold:
                        speaking = True
                        silent_chunks = 0
                        speech_chunks = 1
                        frames.append(data)
                        logger.debug("Speech started (rms=%.4f)", rms)
                else:
                    frames.append(data)
                    if rms >= threshold:
                        silent_chunks = 0
                        speech_chunks += 1
                    else:
                        silent_chunks += 1

                    if silent_chunks >= silence_chunks_limit:
                        if speech_chunks >= min_speech_chunks:
                            logger.debug(
                                "Speech ended (%d chunks, ~%.1fs)",
                                speech_chunks,
                                speech_chunks / chunks_per_sec,
                            )
                            break
                        # Too short — treat as noise and reset
                        frames.clear()
                        speaking = False
                        silent_chunks = 0
                        speech_chunks = 0
        finally:
            self._done.set()  # tell close() we're out of the C code

        if not frames:
            return np.array([], dtype=np.int16)

        raw = b"".join(frames)
        return np.frombuffer(raw, dtype=np.int16)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _rms(data: bytes) -> float:
        """Compute root-mean-square of 16-bit PCM data, normalised to [0, 1]."""
        count = len(data) // 2
        shorts = struct.unpack(f"<{count}h", data)
        sum_sq = sum(s * s for s in shorts)
        return math.sqrt(sum_sq / count) / 32768.0
