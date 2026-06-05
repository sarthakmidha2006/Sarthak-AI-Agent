"""Working-hours availability computation (spec §6).

All scheduling math is timezone-aware. Slots are *generated* in the configured
business timezone (``settings.timezone``) so that working hours are interpreted
locally, but every *overlap* comparison is done in UTC. Bookings and overrides are
stored UTC tz-aware (see :mod:`app.db.models`), and are converted to the business
timezone only for display.

Public surface:

* :class:`Slot` — a tz-aware ``(start, end)`` window with :meth:`Slot.to_dict`.
* :func:`get_available_slots` — enumerate bookable slots across a date range,
  dropping past, booked, and blocked windows.
* :func:`is_slot_available` — validate a single requested start time.
* :func:`_parse_dt` — parse an ISO-8601 string into tz-aware UTC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import Settings
from app.db.models import AvailabilityOverride, Booking

logger = logging.getLogger(__name__)

# Status value treated as an active, conflicting booking.
_CONFIRMED_STATUS = "confirmed"


@dataclass
class Slot:
    """A bookable time window, tz-aware and expressed in the configured timezone.

    Attributes:
        start: Inclusive start of the slot (tz-aware).
        end: Exclusive end of the slot (tz-aware).
    """

    start: datetime
    end: datetime

    def to_dict(self) -> dict:
        """Serialize to ``{"start": iso, "end": iso}`` using ISO-8601 strings."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def _to_utc(value: datetime) -> datetime:
    """Return ``value`` as a tz-aware UTC datetime.

    Naive datetimes are assumed to already be UTC (this should not happen for
    DB-loaded rows, which are tz-aware, but keeps the comparison defensive).
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_dt(value: str) -> datetime:
    """Parse an ISO-8601 string into a tz-aware UTC datetime.

    Accepts values with or without an explicit timezone offset, and tolerates a
    trailing ``Z`` (UTC) designator. Naive values are interpreted in the business
    timezone configured on the settings used by callers; because this helper is
    settings-free it falls back to UTC for naive input only when no caller-supplied
    timezone is available. Callers that need business-tz interpretation pass
    already-localized strings or use :func:`is_slot_available` which localizes.

    Args:
        value: ISO-8601 datetime string.

    Returns:
        A tz-aware datetime normalized to UTC.

    Raises:
        ValueError: If ``value`` is not a parseable ISO-8601 datetime.
    """
    raw = value.strip()
    if raw.endswith("Z") or raw.endswith("z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        # Naive input: treat as UTC at this layer. Callers needing business-tz
        # interpretation (booking) localize before/after this call.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_dt_in_tz(value: str, tz: ZoneInfo) -> datetime:
    """Parse an ISO-8601 string, interpreting naive input in ``tz``, return UTC.

    Args:
        value: ISO-8601 datetime string.
        tz: Business timezone used to localize naive input.

    Returns:
        A tz-aware UTC datetime.

    Raises:
        ValueError: If ``value`` is not parseable.
    """
    raw = value.strip()
    if raw.endswith("Z") or raw.endswith("z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(timezone.utc)


def _overlaps(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    """Return ``True`` if half-open intervals ``[a)`` and ``[b)`` overlap (UTC)."""
    return start_a < end_b and start_b < end_a


def _confirmed_bookings_in_range(
    session,
    *,
    range_start_utc: datetime,
    range_end_utc: datetime,
) -> list[tuple[datetime, datetime]]:
    """Load confirmed bookings overlapping ``[range_start_utc, range_end_utc)``.

    Returns a list of ``(start_utc, end_utc)`` tuples. Querying is broadened with
    a one-day margin and then filtered in Python so we never miss a booking that
    straddles a boundary regardless of how the DB stores tz info.
    """
    margin = timedelta(days=1)
    query = (
        session.query(Booking)
        .filter(Booking.status == _CONFIRMED_STATUS)
        .filter(Booking.end_time > range_start_utc - margin)
        .filter(Booking.start_time < range_end_utc + margin)
    )
    windows: list[tuple[datetime, datetime]] = []
    for booking in query.all():
        windows.append((_to_utc(booking.start_time), _to_utc(booking.end_time)))
    return windows


def _overrides_in_range(
    session,
    *,
    range_start_utc: datetime,
    range_end_utc: datetime,
) -> list[tuple[datetime, datetime]]:
    """Load availability overrides overlapping ``[range_start_utc, range_end_utc)``."""
    margin = timedelta(days=1)
    query = (
        session.query(AvailabilityOverride)
        .filter(AvailabilityOverride.end_time > range_start_utc - margin)
        .filter(AvailabilityOverride.start_time < range_end_utc + margin)
    )
    windows: list[tuple[datetime, datetime]] = []
    for override in query.all():
        windows.append((_to_utc(override.start_time), _to_utc(override.end_time)))
    return windows


def get_available_slots(
    session,
    *,
    settings: Settings,
    date_from: date | None = None,
    date_to: date | None = None,
    duration_minutes: int | None = None,
) -> list[Slot]:
    """Generate bookable working-hours slots across ``[date_from, date_to]``.

    Slots are stepped by ``settings.slot_minutes`` and must fully fit within the
    configured working hours of a working day. A slot is dropped if it:

    * starts in the past (relative to "now"),
    * overlaps any confirmed :class:`~app.db.models.Booking`, or
    * overlaps any :class:`~app.db.models.AvailabilityOverride`.

    Args:
        session: An open SQLAlchemy session for reading bookings/overrides.
        settings: Application settings (timezone, working days/hours, durations).
        date_from: First date to consider (business-tz). Defaults to today.
        date_to: Last date (inclusive) to consider. Defaults to
            ``date_from + settings.booking_horizon_days``.
        duration_minutes: Requested meeting length. Defaults to
            ``settings.booking_default_duration``.

    Returns:
        A chronologically ordered list of available :class:`Slot` objects, each
        expressed in the business timezone.
    """
    tz = ZoneInfo(settings.timezone)
    now_utc = datetime.now(timezone.utc)
    today_local = now_utc.astimezone(tz).date()

    start_date = date_from or today_local
    if date_to is not None:
        end_date = date_to
    else:
        end_date = start_date + timedelta(days=settings.booking_horizon_days)

    if end_date < start_date:
        logger.debug(
            "get_available_slots: date_to %s precedes date_from %s; empty result",
            end_date,
            start_date,
        )
        return []

    duration = int(duration_minutes or settings.booking_default_duration)
    if duration <= 0:
        logger.warning("get_available_slots: non-positive duration %s; empty result", duration)
        return []

    step = int(settings.slot_minutes)
    if step <= 0:
        logger.warning("get_available_slots: non-positive slot_minutes %s; empty result", step)
        return []

    working_days = set(settings.working_days)
    work_start_hour = int(settings.working_hours_start)
    work_end_hour = int(settings.working_hours_end)

    # Pre-load conflicting windows once for the whole range (UTC).
    range_start_utc = datetime.combine(start_date, time.min, tzinfo=tz).astimezone(timezone.utc)
    range_end_utc = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=tz
    ).astimezone(timezone.utc)
    booked = _confirmed_bookings_in_range(
        session, range_start_utc=range_start_utc, range_end_utc=range_end_utc
    )
    overrides = _overrides_in_range(
        session, range_start_utc=range_start_utc, range_end_utc=range_end_utc
    )
    blocked = booked + overrides

    slots: list[Slot] = []
    duration_delta = timedelta(minutes=duration)
    step_delta = timedelta(minutes=step)

    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() not in working_days:
            current_date += timedelta(days=1)
            continue

        day_open = datetime.combine(current_date, time(hour=work_start_hour), tzinfo=tz)
        day_close = datetime.combine(current_date, time(hour=work_end_hour), tzinfo=tz)

        cursor = day_open
        while cursor + duration_delta <= day_close:
            slot_end = cursor + duration_delta
            start_utc = cursor.astimezone(timezone.utc)
            end_utc = slot_end.astimezone(timezone.utc)

            if start_utc < now_utc:
                cursor += step_delta
                continue

            if any(_overlaps(start_utc, end_utc, b_start, b_end) for b_start, b_end in blocked):
                cursor += step_delta
                continue

            slots.append(Slot(start=cursor, end=slot_end))
            cursor += step_delta

        current_date += timedelta(days=1)

    logger.debug(
        "get_available_slots: %d slots from %s to %s (duration=%dm, tz=%s)",
        len(slots),
        start_date,
        end_date,
        duration,
        settings.timezone,
    )
    return slots


def is_slot_available(
    session,
    *,
    settings: Settings,
    start_time: datetime,
    duration_minutes: int,
) -> tuple[bool, str | None]:
    """Validate a specific requested start time.

    Checks, in order, that the requested window: is not in the past, falls on a
    configured working day, fits entirely within working hours, and does not
    overlap a confirmed booking or an availability override. All comparisons are
    done in UTC.

    Args:
        session: An open SQLAlchemy session.
        settings: Application settings.
        start_time: Requested start. May be naive (interpreted in the business
            timezone) or tz-aware (converted to UTC).
        duration_minutes: Requested meeting length in minutes.

    Returns:
        ``(True, None)`` if the slot is bookable, otherwise ``(False, reason)``
        with a human-readable explanation.
    """
    tz = ZoneInfo(settings.timezone)

    if duration_minutes <= 0:
        return False, "Duration must be a positive number of minutes."

    if start_time.tzinfo is None:
        start_local = start_time.replace(tzinfo=tz)
    else:
        start_local = start_time.astimezone(tz)

    start_utc = start_local.astimezone(timezone.utc)
    end_local = start_local + timedelta(minutes=duration_minutes)
    end_utc = start_utc + timedelta(minutes=duration_minutes)

    now_utc = datetime.now(timezone.utc)
    if start_utc < now_utc:
        return False, "That time is in the past."

    if start_local.weekday() not in set(settings.working_days):
        return False, "That day is outside working days."

    day_open = datetime.combine(
        start_local.date(), time(hour=int(settings.working_hours_start)), tzinfo=tz
    )
    day_close = datetime.combine(
        start_local.date(), time(hour=int(settings.working_hours_end)), tzinfo=tz
    )
    if start_local < day_open or end_local > day_close:
        return (
            False,
            (
                "That time is outside working hours "
                f"({settings.working_hours_start:02d}:00–{settings.working_hours_end:02d}:00 "
                f"{settings.timezone})."
            ),
        )

    # Bound the conflict scan to the day containing the request.
    range_start_utc = day_open.astimezone(timezone.utc)
    range_end_utc = day_close.astimezone(timezone.utc)

    booked = _confirmed_bookings_in_range(
        session, range_start_utc=range_start_utc, range_end_utc=range_end_utc
    )
    if any(_overlaps(start_utc, end_utc, b_start, b_end) for b_start, b_end in booked):
        return False, "That time overlaps an existing booking."

    overrides = _overrides_in_range(
        session, range_start_utc=range_start_utc, range_end_utc=range_end_utc
    )
    if any(_overlaps(start_utc, end_utc, o_start, o_end) for o_start, o_end in overrides):
        return False, "That time is blocked and unavailable."

    return True, None
