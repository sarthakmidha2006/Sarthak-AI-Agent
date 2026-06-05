"""``check_availability`` tool implementation (BUILD_SPEC §7.2).

Parses the optional ``date_from`` / ``date_to`` / ``duration_minutes`` arguments
supplied by the model, delegates the actual slot computation to
:func:`app.scheduling.calendar.get_available_slots`, and shapes the result into a
compact, model-friendly dictionary.

The returned payload deliberately caps the number of slots so that the model is
not overwhelmed (and so voice answers stay short). All times are emitted as ISO
8601 strings already converted to the configured timezone by the calendar layer.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.scheduling import calendar

logger = logging.getLogger(__name__)

# Hard cap on the number of slots returned to the model. The spec asks for
# "up to ~12" so that the prompt stays compact and the model does not invent
# slots beyond what was actually offered.
_MAX_SLOTS = 12


def _parse_date(value: Any, *, field: str) -> date | None:
    """Parse an optional ``YYYY-MM-DD`` argument into a :class:`datetime.date`.

    ``None`` / empty input returns ``None`` (meaning "use the calendar default").
    A non-empty value that cannot be parsed raises :class:`ValueError` so that the
    dispatcher can surface a clean ``{"error": ...}`` to the model.
    """

    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a 'YYYY-MM-DD' string")
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid 'YYYY-MM-DD' date") from exc


def _parse_duration(value: Any, *, settings: Settings) -> int:
    """Parse the optional ``duration_minutes`` argument.

    Falls back to ``settings.booking_default_duration`` when omitted. Invalid or
    non-positive values raise :class:`ValueError`.
    """

    if value is None or value == "":
        return settings.booking_default_duration
    try:
        duration = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_minutes must be an integer") from exc
    if duration <= 0:
        raise ValueError("duration_minutes must be a positive integer")
    return duration


def check_availability(arguments: dict, *, session: Session, settings: Settings) -> dict:
    """Return open meeting slots for an optional date range and duration.

    Parameters
    ----------
    arguments:
        The decoded tool-call arguments. Recognised keys (all optional):
        ``date_from`` (``"YYYY-MM-DD"``), ``date_to`` (``"YYYY-MM-DD"``) and
        ``duration_minutes`` (positive int).
    session:
        Active SQLAlchemy session used to read existing bookings/overrides.
    settings:
        Application settings (timezone, working hours, defaults).

    Returns
    -------
    dict
        ``{"timezone": str, "duration_minutes": int,
        "slots": [{"start": iso, "end": iso}, ...], "count": int}``.
    """

    args = arguments or {}
    date_from = _parse_date(args.get("date_from"), field="date_from")
    date_to = _parse_date(args.get("date_to"), field="date_to")
    duration_minutes = _parse_duration(args.get("duration_minutes"), settings=settings)

    logger.debug(
        "check_availability: date_from=%s date_to=%s duration=%s",
        date_from,
        date_to,
        duration_minutes,
    )

    slots = calendar.get_available_slots(
        session,
        settings=settings,
        date_from=date_from,
        date_to=date_to,
        duration_minutes=duration_minutes,
    )

    limited = slots[:_MAX_SLOTS]
    payload = {
        "timezone": settings.timezone,
        "duration_minutes": duration_minutes,
        "slots": [slot.to_dict() for slot in limited],
        "count": len(limited),
    }
    logger.debug(
        "check_availability: found %d slot(s), returning %d",
        len(slots),
        len(limited),
    )
    return payload
