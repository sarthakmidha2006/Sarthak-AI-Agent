"""Public API request/response models (spec §13).

Pydantic v2 models that define the wire contract for the FastAPI routes:
``/chat``, ``/voice``, ``/availability``, and ``/book``. These are the only
schemas the HTTP layer validates against; internal layers use their own
dataclasses (see :mod:`app.rag.schemas`).

``EmailStr`` requires the ``email-validator`` package (declared in
requirements).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

# E.164: "+" followed by 8–15 digits, first digit non-zero.
_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")


class HistoryTurn(BaseModel):
    """A single prior conversational turn supplied by the client."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    history: list[HistoryTurn] | None = None


class Citation(BaseModel):
    """A single grounding citation surfaced alongside an answer."""

    n: int
    title: str
    source_type: str
    url: str | None = None
    snippet: str


class ToolCallView(BaseModel):
    """A tool invocation and its result, as exposed to API clients."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    """Response body for ``POST /chat``."""

    answer: str
    session_id: str
    citations: list[Citation]
    tool_calls: list[ToolCallView]
    injection_flagged: bool
    grounded: bool | None
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class VoiceTextRequest(BaseModel):
    """JSON request body for ``POST /voice`` (text-in path)."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    speak: bool = True


class VoiceResponse(BaseModel):
    """Response body for ``POST /voice`` (both audio and text paths)."""

    answer: str
    session_id: str
    transcript: str | None = None
    audio_base64: str | None = None
    audio_format: str = "mp3"
    citations: list[Citation]
    tool_calls: list[ToolCallView]
    injection_flagged: bool
    latency_ms: float


class SlotView(BaseModel):
    """An available scheduling slot rendered as ISO-8601 start/end strings."""

    start: str
    end: str


class AvailabilityResponse(BaseModel):
    """Response body for ``GET /availability``."""

    timezone: str
    duration_minutes: int
    slots: list[SlotView]
    count: int


class BookRequest(BaseModel):
    """Request body for ``POST /book``."""

    name: str = Field(min_length=1)
    email: EmailStr
    start_time: str
    duration_minutes: int | None = None
    topic: str | None = None


class BookResponse(BaseModel):
    """Response body for ``POST /book``."""

    status: str
    message: str
    booking_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    timezone: str | None = None
    alternatives: list[SlotView] | None = None


class CallMeRequest(BaseModel):
    """Request body for ``POST /call-me`` (outbound AI callback)."""

    name: str | None = Field(default=None, max_length=120)
    phone: str = Field(min_length=8, max_length=20)

    @field_validator("phone")
    @classmethod
    def _validate_e164(cls, v: str) -> str:
        v = v.strip()
        if not _E164_RE.match(v):
            raise ValueError("phone must be E.164 formatted, e.g. +919876543210")
        return v

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class CallMeResponse(BaseModel):
    """Response body for ``POST /call-me``."""

    success: bool
    message: str | None = None
    call_id: str | None = None
