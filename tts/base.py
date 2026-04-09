"""Abstract base class for TTS engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TTSResult:
    """Holds synthesised audio and its sample rate."""
    audio: np.ndarray  # int16 PCM
    sample_rate: int


class BaseTTS(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model weights / warm up the engine."""

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """Convert *text* to speech.  Returns int16 PCM audio."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native sample rate of this engine's output."""
