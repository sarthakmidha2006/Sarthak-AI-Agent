"""Vapi telephony bridge -- OpenAI-compatible custom-LLM endpoint (``/vapi``).

This is the *only* new component needed to give the persona a phone number. The
division of labour is deliberate:

* **Vapi (cloud)** owns everything voice: the inbound phone number, speech-to-text,
  text-to-speech, turn-taking and barge-in/interruptions. Each user turn it issues
  an OpenAI-style ``POST /vapi/chat/completions`` carrying the full running
  transcript.
* **This backend** owns the *thinking*. The adapter maps Vapi's ``messages`` onto
  the SHARED :class:`~app.brain.persona.PersonaBrain` (``channel="voice"``) and
  streams the answer back as OpenAI ``chat.completion.chunk`` SSE events.

Because the brain already runs retrieval, grounding, and the scheduling tool loop
(``check_availability`` / ``book_meeting``), booking and grounded answers work with
**zero** Vapi-side tool/function configuration -- the caller simply hears the
brain's spoken reply. The persona brain, retrieval, grounding, and scheduling
stack are reused unchanged; nothing here touches them.

Conversation continuity for the model comes from Vapi resending the full history
each turn; the per-call ``conversation_id`` is best-effort (derived from Vapi's
``call.id`` when present) and only groups our own DB logs.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_brain, get_settings_dep
from app.config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.brain.persona import PersonaBrain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vapi", tags=["vapi"])

# Spoken when we have no usable user utterance or the brain fails -- kept short so
# it reads naturally over the phone and never leaks internals to the caller.
_FALLBACK = "Sorry, I didn't catch that — could you say it again?"
_GREETING_PROMPT = "Hi! Ask me about my background or projects, or say 'book a meeting'."


def _verify_secret(
    authorization: str | None, x_vapi_secret: str | None, settings: Settings
) -> None:
    """Validate the shared secret Vapi forwards. No-op when ``vapi_secret`` is unset.

    Accepts the secret either as an ``Authorization: Bearer <secret>`` header
    (Vapi custom-LLM credential) or the ``x-vapi-secret`` header (Vapi server
    messages). Leaving ``VAPI_SECRET`` empty disables auth -- convenient for local
    testing, but set it in any deployment Vapi can reach.
    """
    expected = settings.vapi_secret
    if not expected:
        return
    presented: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    presented = presented or x_vapi_secret
    if presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid vapi secret"
        )


def _split_messages(messages: list) -> tuple[str, list[dict]]:
    """Split Vapi's OpenAI ``messages`` into ``(latest_user_query, prior_history)``.

    System and tool messages are dropped (the brain builds its own system prompt).
    The most recent ``user`` message is the query to answer; everything before it
    becomes the history passed to the brain so follow-ups retain context.
    """
    convo = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]
    last_user = next(
        (i for i in range(len(convo) - 1, -1, -1) if convo[i]["role"] == "user"), None
    )
    if last_user is None:
        return "", []
    return convo[last_user]["content"], convo[:last_user]


def _sentences(text: str) -> list[str]:
    """Split ``text`` into speakable chunks so Vapi's TTS can start on sentence 1."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.findall(r"\S.*?(?:[.!?](?=\s|$)|\n|$)", text, re.S)]
    return [p for p in parts if p] or [text]


def _chunk_evt(
    cid: str, created: int, model: str, delta: dict, finish: str | None = None
) -> str:
    """Render one OpenAI ``chat.completion.chunk`` SSE event line."""
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/completions", summary="Vapi custom-LLM bridge to PersonaBrain")
async def vapi_chat_completions(
    request: Request,
    brain: "PersonaBrain" = Depends(get_brain),
    settings: Settings = Depends(get_settings_dep),
    authorization: str | None = Header(default=None),
    x_vapi_secret: str | None = Header(default=None),
):
    """Answer one phone turn via the shared brain, OpenAI-chat-completions style.

    Honours ``stream`` (default ``True``): when streaming, the answer is emitted as
    ``chat.completion.chunk`` SSE events terminated by ``data: [DONE]``; otherwise a
    single ``chat.completion`` JSON object is returned. The brain is always invoked
    on the ``"voice"`` channel.
    """
    _verify_secret(authorization, x_vapi_secret, settings)

    body = await request.json()
    messages = body.get("messages") or []
    model = body.get("model") or "persona-brain"
    stream = bool(body.get("stream", True))
    # Best-effort conversation id for our own DB logs. The model's continuity comes
    # from Vapi resending full history each turn, so a fallback uuid is harmless.
    conv_id = (
        (body.get("call") or {}).get("id")
        or (body.get("metadata") or {}).get("call_id")
        or uuid.uuid4().hex
    )
    query, history = _split_messages(messages)

    async def run() -> str:
        if not query.strip():
            return _GREETING_PROMPT
        try:
            result = await brain.answer(
                query,
                channel="voice",
                history=history or None,
                conversation_id=conv_id,
            )
            return (result.answer or "").strip() or _FALLBACK
        except Exception:  # noqa: BLE001 - never surface a 500 into a live phone call
            logger.exception("vapi: brain.answer failed (conv=%s)", conv_id)
            return _FALLBACK

    if not stream:
        answer = await run()
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def sse() -> AsyncIterator[str]:
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        yield _chunk_evt(cid, created, model, {"role": "assistant"})
        answer = await run()
        for piece in _sentences(answer):
            yield _chunk_evt(cid, created, model, {"content": piece + " "})
        yield _chunk_evt(cid, created, model, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@router.post("/events", summary="Vapi server webhook (optional: lifecycle logs)")
async def vapi_events(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    x_vapi_secret: str | None = Header(default=None),
):
    """Receive Vapi server messages (e.g. ``end-of-call-report``) and ack them.

    Optional and side-effect-free: it just logs the message type so call lifecycle
    and cost can be observed. Secured by the same shared secret as the bridge.
    """
    _verify_secret(None, x_vapi_secret, settings)
    body = await request.json()
    message = body.get("message") or {}
    logger.info("vapi event: type=%s", message.get("type"))
    return {"received": True}
