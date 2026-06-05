"""Chat channel route -- ``POST /chat``.

Accepts a :class:`~app.models.api_schemas.ChatRequest`, runs it through the
shared :class:`~app.brain.persona.PersonaBrain` on the ``"chat"`` channel, and
maps the resulting :class:`~app.brain.persona.BrainResponse` onto a
:class:`~app.models.api_schemas.ChatResponse` (spec sections 13 and 14.2).

The route is intentionally thin: all retrieval, tool-calling, grounding and
persistence logic lives inside the brain so that the chat and voice channels
share identical behaviour.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_brain
from app.models.api_schemas import ChatRequest, ChatResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.brain.persona import BrainResponse, PersonaBrain

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _to_chat_response(result: "BrainResponse", session_id: str) -> ChatResponse:
    """Map a :class:`~app.brain.persona.BrainResponse` to a :class:`ChatResponse`.

    The brain returns citations and tool calls as plain dictionaries; pydantic
    validates and coerces them into the typed response models.
    """

    return ChatResponse(
        answer=result.answer,
        session_id=session_id,
        citations=result.citations,
        tool_calls=result.tool_calls,
        injection_flagged=result.injection_flagged,
        grounded=result.grounded,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


@router.post("/chat", response_model=ChatResponse, summary="Chat with the persona")
async def chat(
    payload: ChatRequest,
    brain: "PersonaBrain" = Depends(get_brain),
) -> ChatResponse:
    """Answer a text question via the shared brain on the chat channel.

    The ``session_id`` from the request is reused as the conversation id so the
    brain can attach the turn to an existing conversation; if absent, a fresh
    uuid4 hex string is generated and returned to the client for continuity.
    """

    session_id = payload.session_id or uuid.uuid4().hex
    history = (
        [{"role": turn.role, "content": turn.content} for turn in payload.history]
        if payload.history
        else None
    )

    try:
        result = await brain.answer(
            payload.message,
            channel="chat",
            history=history,
            conversation_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 502 to the client
        logger.exception("Chat request failed for session %s", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate a response.",
        ) from exc

    # The brain owns the canonical conversation id; honour it if it differs.
    resolved_session_id = result.conversation_id or session_id
    return _to_chat_response(result, resolved_session_id)
