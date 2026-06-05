"""``book_meeting`` tool implementation (BUILD_SPEC §7.3).

Validates the requested attendee details and start time, checks the slot against
existing bookings / blocked windows, and either creates a confirmed
:class:`app.db.models.Booking` row or returns the slot as unavailable together
with up to three alternative slots.

This function backs **both** the brain's tool path (channel ``"chat"`` /
``"voice"``) and the REST ``POST /book`` route (channel ``"api"``) — the only
difference is the ``channel`` argument that is persisted on the booking.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Booking
from app.scheduling import calendar

logger = logging.getLogger(__name__)

# Pragmatic email syntax check. We deliberately avoid full RFC 5322 — the goal is
# to reject obviously malformed addresses before persisting a booking, not to be
# a complete validator. EmailStr in the pydantic API layer provides the stricter
# guarantee on the REST surface; this keeps the tool path self-contained.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Number of alternative slots to suggest when the requested time is unavailable.
_MAX_ALTERNATIVES = 3


def _require_str(value: Any, *, field: str) -> str:
    """Return a non-empty, stripped string or raise :class:`ValueError`."""

    if value is None:
        raise ValueError(f"{field} is required")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _validate_email(value: Any) -> str:
    """Validate and normalise an email address (basic regex per spec)."""

    email = _require_str(value, field="email")
    if not _EMAIL_RE.match(email):
        raise ValueError(f"'{email}' is not a valid email address")
    return email


def _resolve_duration(value: Any, *, settings: Settings) -> int:
    """Resolve the meeting duration, defaulting to the configured value."""

    if value is None or value == "":
        return settings.booking_default_duration
    try:
        duration = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_minutes must be an integer") from exc
    if duration <= 0:
        raise ValueError("duration_minutes must be a positive integer")
    return duration


def _suggest_alternatives(
    session: Session, *, settings: Settings, duration_minutes: int, around: datetime
) -> list[dict]:
    """Return up to ``_MAX_ALTERNATIVES`` open slots near ``around``.

    ``around`` is a tz-aware UTC datetime. Availability is searched from that
    day forward across the configured booking horizon.
    """

    tz = ZoneInfo(settings.timezone)
    local_day = around.astimezone(tz).date()
    slots = calendar.get_available_slots(
        session,
        settings=settings,
        date_from=local_day,
        date_to=None,
        duration_minutes=duration_minutes,
    )
    return [slot.to_dict() for slot in slots[:_MAX_ALTERNATIVES]]


def book_meeting(arguments: dict, *, session: Session, settings: Settings, channel: str) -> dict:
    """Book a meeting or report the requested slot as unavailable.

    Parameters
    ----------
    arguments:
        Decoded tool-call arguments: ``name`` (str, required), ``email`` (str,
        required), ``start_time`` (ISO-8601 str, required), ``duration_minutes``
        (int, optional) and ``topic`` (str, optional).
    session:
        Active SQLAlchemy session; the booking is committed on success.
    settings:
        Application settings (timezone, working hours, defaults).
    channel:
        Origin channel persisted on the booking: ``"chat"`` | ``"voice"`` |
        ``"api"``.

    Returns
    -------
    dict
        On success::

            {"status": "confirmed", "booking_id": str, "name": str,
             "email": str, "start_time": iso, "end_time": iso,
             "topic": str | None, "timezone": str}

        When the slot cannot be booked::

            {"status": "unavailable", "reason": str,
             "alternatives": [{"start": iso, "end": iso}, ...]}
    """

    args = arguments or {}
    name = _require_str(args.get("name"), field="name")
    email = _validate_email(args.get("email"))
    start_raw = _require_str(args.get("start_time"), field="start_time")
    duration_minutes = _resolve_duration(args.get("duration_minutes"), settings=settings)

    topic_value = args.get("topic")
    topic: str | None
    if topic_value is None:
        topic = None
    elif isinstance(topic_value, str):
        topic = topic_value.strip() or None
    else:
        raise ValueError("topic must be a string")

    # Parse the requested start into tz-aware UTC. A naive ISO string (no offset)
    # is interpreted in the business timezone — a model/client that says
    # "2026-06-10T14:00:00" means 2pm local, not 2pm UTC (BUILD_SPEC §6). Offset-
    # bearing and trailing-"Z" inputs round-trip to the same UTC value. Invalid
    # input raises ValueError, which the dispatcher converts into {"error": ...}.
    start_utc = calendar._parse_dt_in_tz(start_raw, ZoneInfo(settings.timezone))

    ok, reason = calendar.is_slot_available(
        session,
        settings=settings,
        start_time=start_utc,
        duration_minutes=duration_minutes,
    )
    if not ok:
        alternatives = _suggest_alternatives(
            session,
            settings=settings,
            duration_minutes=duration_minutes,
            around=start_utc,
        )
        logger.info(
            "book_meeting: slot %s unavailable (%s); offering %d alternative(s)",
            start_utc.isoformat(),
            reason,
            len(alternatives),
        )
        return {
            "status": "unavailable",
            "reason": reason or "The requested time is not available.",
            "alternatives": alternatives,
        }

    end_utc = start_utc + timedelta(minutes=duration_minutes)
    booking = Booking(
        name=name,
        email=email,
        start_time=start_utc,
        end_time=end_utc,
        topic=topic,
        status="confirmed",
        channel=channel,
    )

    try:
        session.add(booking)
        session.commit()
        session.refresh(booking)
    except Exception:  # pragma: no cover - defensive; surfaced as {"error": ...}
        session.rollback()
        logger.exception("book_meeting: failed to persist booking")
        raise

    tz = ZoneInfo(settings.timezone)
    start_local = start_utc.astimezone(tz)
    end_local = end_utc.astimezone(tz)

    logger.info(
        "book_meeting: confirmed booking %s for %s at %s (channel=%s)",
        booking.id,
        email,
        start_local.isoformat(),
        channel,
    )
    return {
        "status": "confirmed",
        "booking_id": booking.id,
        "name": name,
        "email": email,
        "start_time": start_local.isoformat(),
        "end_time": end_local.isoformat(),
        "topic": topic,
        "timezone": settings.timezone,
    }
