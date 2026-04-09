"""Console TTS — prints text to stdout instead of synthesising speech."""

from __future__ import annotations

import numpy as np

from .base import BaseTTS, TTSResult


class ConsoleTTS(BaseTTS):
    """No-op TTS that prints the answer to the terminal.
    Useful for testing the pipeline without TTS models or audio hardware."""

    def load(self) -> None:
        pass

    def synthesize(self, text: str) -> TTSResult:
        print(f"🔊 {text}", end="", flush=True)
        return TTSResult(audio=np.array([], dtype=np.int16), sample_rate=16000)

    @property
    def sample_rate(self) -> int:
        return 16000
