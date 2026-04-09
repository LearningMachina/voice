"""Fake learning-hub API server.

Implements the subset of the learning-hub API that the voice app uses:
  GET  /api/health
  POST /api/ask
  POST /api/ask/stream

Answers are canned responses — no LLM or database required.
Run standalone:  python -m fake_hub.server [--port 3000]
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import argparse
from datetime import datetime, timezone

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

CANNED_ANSWERS = [
    (
        "A GPIO (General-Purpose Input/Output) pin is a digital signal pin on a "
        "microcontroller or single-board computer. It can be configured as either "
        "an input to read sensors and buttons, or an output to control LEDs and "
        "motors. GPIO pins typically operate at 3.3V or 5V logic levels."
    ),
    (
        "PWM stands for Pulse Width Modulation. It's a technique where you "
        "rapidly switch a digital pin on and off to simulate an analogue voltage. "
        "By changing the duty cycle — the percentage of time the signal is high — "
        "you can control things like LED brightness or motor speed."
    ),
    (
        "An LED is a Light Emitting Diode. It's a semiconductor device that "
        "emits light when current flows through it. LEDs are energy-efficient, "
        "have a long lifespan, and come in many colours. To use one with a "
        "microcontroller, you connect it through a current-limiting resistor."
    ),
    (
        "I²C, or Inter-Integrated Circuit, is a serial communication protocol "
        "that lets multiple devices talk over just two wires: SDA for data and "
        "SCL for the clock. It's commonly used for sensors, displays, and other "
        "peripherals on embedded systems."
    ),
    (
        "A breadboard is a rectangular plastic board with a grid of holes "
        "connected by metal strips underneath. It lets you build and test "
        "electronic circuits without soldering. Components and jumper wires "
        "push into the holes to make temporary connections."
    ),
]

_answer_index = 0
_conversations: dict[str, list[dict]] = {}


def _next_answer(question: str) -> tuple[str, str]:
    """Return (answer_text, conversation_id) cycling through canned answers."""
    global _answer_index
    answer = CANNED_ANSWERS[_answer_index % len(CANNED_ANSWERS)]
    _answer_index += 1
    conv_id = f"conv_{uuid.uuid4().hex[:8]}"
    return answer, conv_id


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "version": "0.1.0-fake",
        "llm": {"status": "connected", "provider": "fake", "model": "canned"},
        "db": {"status": "connected"},
    })


async def ask(request: Request) -> JSONResponse:
    body = await request.json()
    question = body.get("question", "")
    conv_id = body.get("conversation_id")

    if not question:
        return JSONResponse(
            {"error": {"code": "VALIDATION_ERROR", "message": "Field 'question' is required"}},
            status_code=400,
        )

    answer, new_conv_id = _next_answer(question)
    conv_id = conv_id or new_conv_id

    return JSONResponse({
        "answer": answer,
        "conversation_id": conv_id,
        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
        "model": "canned",
        "usage": {"prompt_tokens": len(question.split()), "completion_tokens": len(answer.split())},
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


async def ask_stream(request: Request) -> StreamingResponse:
    body = await request.json()
    question = body.get("question", "")
    conv_id = body.get("conversation_id")

    if not question:
        return JSONResponse(
            {"error": {"code": "VALIDATION_ERROR", "message": "Field 'question' is required"}},
            status_code=400,
        )

    answer, new_conv_id = _next_answer(question)
    conv_id = conv_id or new_conv_id

    async def event_stream():
        # Split answer into word-sized chunks to simulate streaming
        words = answer.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield f"event: chunk\ndata: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.05)  # simulate generation delay

        done_data = {
            "conversation_id": conv_id,
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "model": "canned",
            "usage": {"prompt_tokens": len(question.split()), "completion_tokens": len(words)},
        }
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/ask", ask, methods=["POST"]),
    Route("/api/ask/stream", ask_stream, methods=["POST"]),
]

app = Starlette(routes=routes)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Fake learning-hub API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    print(f"🤖 Fake learning-hub running on http://{args.host}:{args.port}")
    print("   Endpoints: /api/health, /api/ask, /api/ask/stream")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
