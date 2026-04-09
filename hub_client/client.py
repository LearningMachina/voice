"""Async HTTP client for the learning-hub REST API with SSE streaming support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


@dataclass
class AskResponse:
    answer: str
    conversation_id: str
    message_id: str = ""
    model: str = ""


class HubClient:
    """Client for the learning-hub ``/api/*`` endpoints."""

    def __init__(self, cfg: Config) -> None:
        self._base_url = cfg.hub_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0),
        )
        logger.info("Hub client ready (base_url=%s)", self._base_url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- health ----------------------------------------------------------------

    async def health(self) -> dict:
        assert self._client is not None
        resp = await self._client.get("/api/health")
        resp.raise_for_status()
        return resp.json()

    # -- ask (synchronous response) -------------------------------------------

    async def ask(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AskResponse:
        assert self._client is not None
        payload: dict = {"question": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        resp = await self._client.post("/api/ask", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return AskResponse(
            answer=data["answer"],
            conversation_id=data["conversation_id"],
            message_id=data.get("message_id", ""),
            model=data.get("model", ""),
        )

    # -- ask (SSE streaming) --------------------------------------------------

    async def ask_stream(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer text chunks as they arrive via SSE.

        The final ``done`` event is consumed internally and updates
        ``self.last_conversation_id``.
        """
        assert self._client is not None
        payload: dict = {"question": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with self._client.stream(
            "POST",
            "/api/ask/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            current_event = ""
            async for line in resp.aiter_lines():
                line = line.rstrip("\r\n")

                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue

                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "chunk" and "text" in data:
                        yield data["text"]
                    elif current_event == "done":
                        cid = data.get("conversation_id", "")
                        logger.debug("Stream done (conversation_id=%s)", cid)
                        self._last_conversation_id = cid
                    current_event = ""

                # blank lines are SSE event delimiters — just skip

    @property
    def last_conversation_id(self) -> str | None:
        return getattr(self, "_last_conversation_id", None)
