"""LearningMachina Voice — main entry point.

Orchestrates the listen → STT → ask → TTS → speak loop.
"""

from __future__ import annotations

import asyncio
import logging
import re
import signal
import sys

_MISSING: list[str] = []
for _mod in ("numpy", "httpx", "dotenv"):
    try:
        __import__(_mod)
    except ImportError:
        _MISSING.append(_mod)
if _MISSING:
    print(
        "ERROR: Missing required packages. "
        "Install dependencies first:\n\n"
        "  pip install -r requirements.txt\n"
    )
    sys.exit(1)

import numpy as np
import httpx

from config import load_config, Config
from stt import create_stt_engine
from stt.whisper_stt import BaseSTT
from tts import create_tts_engine
from tts.base import BaseTTS
from hub_client import HubClient
from audio.capture import MicCapture
from audio.playback import Speaker

logger = logging.getLogger("voice")

# Sentence boundary regex — split on . ! ? followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split *text* into complete sentences and a remaining buffer.

    Returns (sentences, remaining).  Sentences shorter than 10 chars are
    kept in the buffer to avoid choppy TTS calls.
    """
    parts = _SENTENCE_RE.split(text)
    if len(parts) <= 1:
        return [], text

    sentences = parts[:-1]
    remaining = parts[-1]

    # Merge very short sentences into the next one
    merged: list[str] = []
    buf = ""
    for s in sentences:
        buf = (buf + " " + s).strip() if buf else s
        if len(buf) >= 10:
            merged.append(buf)
            buf = ""
    if buf:
        remaining = (buf + " " + remaining).strip()

    return merged, remaining


class VoiceApp:
    """Main application that ties all components together."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stt: BaseSTT = create_stt_engine(cfg)
        self.tts: BaseTTS = create_tts_engine(cfg)
        self.hub = HubClient(cfg)
        self.mic: MicCapture | None = None
        self.speaker: Speaker | None = None
        self.conversation_id: str | None = None
        self._use_audio = cfg.stt_engine != "console"

    async def start(self) -> None:
        """Load models and open audio devices."""
        logger.info("Initialising voice app …")

        # Load STT
        logger.info("Loading STT engine: %s", self.cfg.stt_engine)
        await asyncio.to_thread(self.stt.load)

        # Load TTS
        logger.info("Loading TTS engine: %s", self.cfg.tts_engine)
        await asyncio.to_thread(self.tts.load)

        # Open hub client
        await self.hub.open()

        # Open audio devices (skip in console mode)
        if self._use_audio:
            self.mic = MicCapture(self.cfg)
            self.mic.open()

        if self.cfg.tts_engine != "console":
            self.speaker = Speaker(self.cfg)
            self.speaker.open()

        logger.info("Voice app ready!")

    async def stop(self) -> None:
        """Clean up resources.  Order matters: audio streams first."""
        if self.mic:
            self.mic.close()
            self.mic = None
        if self.speaker:
            self.speaker.close()
            self.speaker = None
        await self.hub.close()
        logger.info("Voice app stopped.")

    async def run(self) -> None:
        """Main conversation loop."""
        if self._use_audio:
            input_mode = f"🎤 Microphone (Whisper {self.cfg.whisper_model})"
        else:
            input_mode = "⌨️  Keyboard (console)"
        output_mode = self.cfg.tts_engine if self.cfg.tts_engine != "console" else "console (text)"

        print("\n╔══════════════════════════════════════════╗")
        print("║   LearningMachina Voice — Ready!         ║")
        print("╚══════════════════════════════════════════╝")
        print(f"  Input:  {input_mode}")
        print(f"  Output: {output_mode}")
        print(f"  Hub:    {self.cfg.hub_url}")
        print("  Press Ctrl+C to exit.\n")

        try:
            while True:
                await self._turn()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")

    async def _turn(self) -> None:
        """Run one listen → transcribe → ask → speak turn."""

        # -- 1. Capture speech / read from console --
        if self._use_audio:
            assert self.mic is not None
            print("🎤 Listening … (speak now)", flush=True)
            audio = await asyncio.to_thread(self.mic.record_speech)
            text = await asyncio.to_thread(self.stt.transcribe, audio, self.cfg.sample_rate)
        else:
            text = await asyncio.to_thread(
                self.stt.transcribe,
                np.array([], dtype=np.int16),
                self.cfg.sample_rate,
            )

        if not text.strip():
            return

        if self._use_audio:
            print(f"🎤 You: {text}")

        # -- 2. Stream response from hub --
        print("🤖 ", end="", flush=True)
        sentence_buffer = ""
        full_answer = ""

        try:
            async for chunk in self.hub.ask_stream(text, self.conversation_id):
                full_answer += chunk
                sentence_buffer += chunk

                # Check for complete sentences to speak
                sentences, sentence_buffer = _split_sentences(sentence_buffer)
                for sentence in sentences:
                    if self.cfg.tts_engine == "console":
                        print(sentence, end=" ", flush=True)
                    else:
                        await self._speak(sentence)
                        print(sentence, end=" ", flush=True)

        except httpx.HTTPError as exc:
            logger.warning("Streaming failed (%s), falling back to sync ask", exc)
            response = await self.hub.ask(text, self.conversation_id)
            full_answer = response.answer
            sentence_buffer = full_answer
            self.conversation_id = response.conversation_id

        # Speak any remaining buffered text
        if sentence_buffer.strip():
            if self.cfg.tts_engine == "console":
                print(sentence_buffer, flush=True)
            else:
                await self._speak(sentence_buffer)
                print(sentence_buffer, flush=True)

        print()  # newline after answer

        # Update conversation context
        if self.hub.last_conversation_id:
            self.conversation_id = self.hub.last_conversation_id

    async def _speak(self, text: str) -> None:
        """Synthesise text and play through the speaker."""
        result = await asyncio.to_thread(self.tts.synthesize, text)
        if self.speaker and len(result.audio) > 0:
            audio = result.audio
            if result.sample_rate != self.cfg.sample_rate:
                logger.debug(
                    "Resampling %d→%d Hz (%d samples)",
                    result.sample_rate, self.cfg.sample_rate, len(audio),
                )
                audio = self._resample(audio, result.sample_rate, self.cfg.sample_rate)
            await asyncio.to_thread(self.speaker.play, audio)
        elif self.speaker:
            logger.warning("TTS returned empty audio for: %r", text[:60])

    @staticmethod
    def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear resampling (for rate conversion between TTS and output)."""
        if from_rate == to_rate:
            return audio
        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(np.int16)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = VoiceApp(cfg)

    async def _run() -> None:
        await app.start()
        try:
            await app.run()
        finally:
            await app.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Audio streams already closed in app.stop() (finally block).
        # Belt-and-suspenders: close again in case stop() was skipped.
        if app.mic:
            app.mic.close()
        if app.speaker:
            app.speaker.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
