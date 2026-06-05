"""SQLAlchemy 2.0 ORM models (spec §5.2).

All datetime columns are timezone-aware and stored in UTC; conversions to the
configured display timezone happen only in the scheduling/display layers. Primary
keys are uuid4 hex strings. JSON columns use :class:`sqlalchemy.JSON`.

Tables:
    * :class:`Booking` — confirmed/cancelled meetings.
    * :class:`AvailabilityOverride` — blocked windows (PTO/holidays).
    * :class:`Conversation` — a chat/voice session.
    * :class:`Message` — individual turns within a conversation.
    * :class:`QueryLog` — one row per brain answer (latency + eval signals).
    * :class:`EvalResult` — persisted evaluation metric rows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _uuid_hex() -> str:
    """Return a new uuid4 hex string suitable for a primary key."""

    return uuid.uuid4().hex


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


class Booking(Base):
    """A scheduled meeting.

    Times are persisted in UTC. ``status`` is ``"confirmed"`` or
    ``"cancelled"``; ``channel`` records the origin (``"chat"``/``"voice"``/
    ``"api"``).
    """

    __tablename__ = "bookings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="confirmed")
    channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""

        return (
            f"<Booking id={self.id!r} name={self.name!r} "
            f"start_time={self.start_time!r} status={self.status!r} "
            f"channel={self.channel!r}>"
        )


class AvailabilityOverride(Base):
    """A blocked window during which no bookings may be made (PTO/holidays)."""

    __tablename__ = "availability_overrides"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class Conversation(Base):
    """A chat or voice session that groups a series of :class:`Message` turns."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class Message(Base):
    """A single turn within a :class:`Conversation`.

    ``role`` is one of ``"user"``, ``"assistant"``, ``"tool"``, ``"system"``.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class QueryLog(Base):
    """One row per brain answer; powers latency analysis and evaluation.

    Captures the query/answer pair, retrieval provenance (chunk ids, citations),
    any tool calls, token usage, latency totals/breakdowns, and the security
    signals (injection flag, grounding verdict).
    """

    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")

    retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    latency_ms_total: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    injection_flagged: Mapped[bool] = mapped_column(default=False, nullable=False)
    grounded: Mapped[bool | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class EvalResult(Base):
    """A single persisted evaluation metric value for an eval run."""

    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metric: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
