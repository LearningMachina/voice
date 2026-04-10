"""Fake learning-hub API server.

Implements the subset of the learning-hub API that the voice app uses:
  GET  /api/health
  POST /api/ask
  POST /api/ask/stream

If GEMINI_API_KEY is set in .env / environment, answers come from
Google Gemini.  Otherwise falls back to canned responses.

Run standalone:  python -m fake_hub.server [--port 3000]
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import argparse
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

load_dotenv()
logger = logging.getLogger("fake_hub")

# ---------------------------------------------------------------------------
# Gemini setup (optional)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a friendly and patient learning tutor for the LearningMachina "
    "educational robot. Your students are young learners exploring electronics, "
    "programming, and robotics.\n\n"
    "Guidelines:\n"
    "- Give clear, concise answers suitable for spoken delivery (the answer "
    "will be read aloud by a text-to-speech engine).\n"
    "- Keep answers to 2-4 sentences unless the student asks for more detail.\n"
    "- Use simple language; avoid jargon unless you explain it.\n"
    "- Be encouraging and enthusiastic about learning.\n"
    "- When explaining technical concepts, use real-world analogies.\n"
    "- If a question is unclear, ask the student to rephrase."
)

_gemini_client = None


def _get_gemini_client():
    """Lazily initialise the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client ready (model=%s)", GEMINI_MODEL)
    return _gemini_client


# Conversation history keyed by conversation_id
_conversations: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Canned responses (fallback when no API key)
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


def _next_canned(question: str) -> str:
    """Return the next canned answer (round-robin)."""
    global _answer_index
    answer = CANNED_ANSWERS[_answer_index % len(CANNED_ANSWERS)]
    _answer_index += 1
    return answer


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _build_contents(conv_id: str | None, question: str) -> list[dict]:
    """Build the Gemini contents list from conversation history."""
    history = _conversations.get(conv_id, []) if conv_id else []
    contents = list(history)
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents


def _save_turn(conv_id: str, question: str, answer: str) -> None:
    """Append a Q&A turn to conversation history."""
    if conv_id not in _conversations:
        _conversations[conv_id] = []
    _conversations[conv_id].append({"role": "user", "parts": [{"text": question}]})
    _conversations[conv_id].append({"role": "model", "parts": [{"text": answer}]})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
    backend = "gemini" if GEMINI_API_KEY else "fake"
    return JSONResponse({
        "status": "ok",
        "version": "0.1.0-fake",
        "llm": {
            "status": "connected",
            "provider": backend,
            "model": GEMINI_MODEL if GEMINI_API_KEY else "canned",
        },
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

    conv_id = conv_id or f"conv_{uuid.uuid4().hex[:8]}"

    if GEMINI_API_KEY:
        try:
            from google.genai import types

            client = _get_gemini_client()
            contents = _build_contents(conv_id, question)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                ),
            )
            answer = response.text or ""
            _save_turn(conv_id, question, answer)
        except Exception as exc:
            logger.exception("Gemini call failed, falling back to canned")
            answer = _next_canned(question)
    else:
        answer = _next_canned(question)

    return JSONResponse({
        "answer": answer,
        "conversation_id": conv_id,
        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
        "model": GEMINI_MODEL if GEMINI_API_KEY else "canned",
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

    conv_id = conv_id or f"conv_{uuid.uuid4().hex[:8]}"

    if GEMINI_API_KEY:
        async def gemini_event_stream():
            from google.genai import types

            full_answer = ""
            try:
                client = _get_gemini_client()
                contents = _build_contents(conv_id, question)
                stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7,
                    ),
                )
                for chunk in stream:
                    text = chunk.text or ""
                    if text:
                        full_answer += text
                        yield f"event: chunk\ndata: {json.dumps({'text': text})}\n\n"
            except Exception as exc:
                logger.exception("Gemini streaming failed, sending canned answer")
                full_answer = _next_canned(question)
                words = full_answer.split()
                for i, word in enumerate(words):
                    token = word if i == 0 else " " + word
                    yield f"event: chunk\ndata: {json.dumps({'text': token})}\n\n"
                    await asyncio.sleep(0.05)

            _save_turn(conv_id, question, full_answer)

            done_data = {
                "conversation_id": conv_id,
                "message_id": f"msg_{uuid.uuid4().hex[:8]}",
                "model": GEMINI_MODEL,
                "usage": {
                    "prompt_tokens": len(question.split()),
                    "completion_tokens": len(full_answer.split()),
                },
            }
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

        return StreamingResponse(gemini_event_stream(), media_type="text/event-stream")

    # Fallback: canned responses
    answer = _next_canned(question)

    async def canned_event_stream():
        words = answer.split()
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield f"event: chunk\ndata: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.05)

        done_data = {
            "conversation_id": conv_id,
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "model": "canned",
            "usage": {"prompt_tokens": len(question.split()), "completion_tokens": len(words)},
        }
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    return StreamingResponse(canned_event_stream(), media_type="text/event-stream")


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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fake learning-hub API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    backend = f"Gemini ({GEMINI_MODEL})" if GEMINI_API_KEY else "canned responses"
    print(f"🤖 Fake learning-hub running on http://{args.host}:{args.port}")
    print(f"   Backend: {backend}")
    print("   Endpoints: /api/health, /api/ask, /api/ask/stream")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
