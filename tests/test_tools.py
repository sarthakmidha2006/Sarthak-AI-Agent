"""Tests for the scheduling calendar and the booking/availability tools (§17).

Covers (per spec §17):

* :func:`app.scheduling.calendar.is_slot_available` honours working hours,
  working days, past times, confirmed bookings, and availability overrides.
* :func:`app.scheduling.calendar.get_available_slots` enumerates the correct
  number of working-hours slots and drops past / overridden days.
* :func:`app.tools.booking.book_meeting` confirms a slot, then rejects a
  double-book (returning alternatives), and rejects an invalid email.

All datetime math is anchored to "now" so the suite never goes stale, and a
helper picks the next *future* working day in the configured timezone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.config import Settings
from app.db.models import AvailabilityOverride, Booking
from app.scheduling import calendar
from app.tools.availability import check_availability
from app.tools.booking import book_meeting


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _next_working_day(settings: Settings, *, after_days: int = 0):
    """Return the date of the next strictly-future working day in the business tz.

    Args:
        settings: Application settings (timezone + working days).
        after_days: Minimum number of days ahead to start searching from.
    """

    tz = ZoneInfo(settings.timezone)
    today = datetime.now(tz).date()
    candidate = today + timedelta(days=max(1, after_days))
    working = set(settings.working_days)
    while candidate.weekday() not in working:
        candidate += timedelta(days=1)
    return candidate


def _local_dt(settings: Settings, day, hour: int, minute: int = 0) -> datetime:
    """Build a tz-aware datetime on ``day`` at ``hour:minute`` in the business tz."""

    tz = ZoneInfo(settings.timezone)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)


# --------------------------------------------------------------------------- #
# is_slot_available
# --------------------------------------------------------------------------- #
def test_is_slot_available_accepts_future_working_hours(session, settings: Settings) -> None:
    """A future, in-hours slot on a working day with no conflicts is available."""

    day = _next_working_day(settings)
    start = _local_dt(settings, day, 10, 0)

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=start, duration_minutes=30
    )

    assert ok is True
    assert reason is None


def test_is_slot_available_rejects_past(session, settings: Settings) -> None:
    """A start time in the past is rejected with a clear reason."""

    past = datetime.now(timezone.utc) - timedelta(days=1)

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=past, duration_minutes=30
    )

    assert ok is False
    assert reason is not None
    assert "past" in reason.lower()


def test_is_slot_available_rejects_outside_working_hours(session, settings: Settings) -> None:
    """A slot before the working-hours window is rejected."""

    day = _next_working_day(settings)
    too_early = _local_dt(settings, day, settings.working_hours_start - 1, 0)

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=too_early, duration_minutes=30
    )

    assert ok is False
    assert reason is not None
    assert "working hours" in reason.lower()


def test_is_slot_available_rejects_slot_spilling_past_close(session, settings: Settings) -> None:
    """A slot that starts in-hours but ends after close does not fully fit."""

    day = _next_working_day(settings)
    # Starts at 16:45, a 30-minute meeting would end at 17:15 (> 17:00 close).
    start = _local_dt(settings, day, settings.working_hours_end - 1, 45)

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=start, duration_minutes=30
    )

    assert ok is False
    assert reason is not None
    assert "working hours" in reason.lower()


def test_is_slot_available_rejects_weekend(session, settings: Settings) -> None:
    """A slot on a non-working day (Saturday) is rejected."""

    tz = ZoneInfo(settings.timezone)
    today = datetime.now(tz).date()
    # Find the next future Saturday (weekday 5), which is not in working_days.
    candidate = today + timedelta(days=1)
    while candidate.weekday() != 5:
        candidate += timedelta(days=1)
    saturday = _local_dt(settings, candidate, 10, 0)

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=saturday, duration_minutes=30
    )

    assert ok is False
    assert reason is not None
    assert "working day" in reason.lower()


def test_is_slot_available_rejects_overlapping_booking(session, settings: Settings) -> None:
    """A slot overlapping an existing confirmed booking is rejected."""

    day = _next_working_day(settings)
    start_local = _local_dt(settings, day, 11, 0)
    start_utc = start_local.astimezone(timezone.utc)

    session.add(
        Booking(
            name="Existing",
            email="existing@example.com",
            start_time=start_utc,
            end_time=start_utc + timedelta(minutes=30),
            status="confirmed",
            channel="api",
        )
    )
    session.commit()

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=start_local, duration_minutes=30
    )

    assert ok is False
    assert reason is not None
    assert "booking" in reason.lower()


def test_is_slot_available_ignores_cancelled_booking(session, settings: Settings) -> None:
    """A cancelled booking does not block the slot (only confirmed conflict)."""

    day = _next_working_day(settings)
    start_local = _local_dt(settings, day, 11, 0)
    start_utc = start_local.astimezone(timezone.utc)

    session.add(
        Booking(
            name="Cancelled",
            email="cancelled@example.com",
            start_time=start_utc,
            end_time=start_utc + timedelta(minutes=30),
            status="cancelled",
            channel="api",
        )
    )
    session.commit()

    ok, reason = calendar.is_slot_available(
        session, settings=settings, start_time=start_local, duration_minutes=30
    )

    assert ok is True
    assert reason is None


def test_is_slot_available_rejects_override(session, settings: Settings) -> None:
    """A slot overlapping an availability override (PTO) is rejected."""

    day = _next_working_day(settings)
    block_start = _local_dt(settings, day, 9, 0).astimezone(timezone.utc)
    block_end = _local_dt(settings, day, 17, 0).astimezone(timezone.utc)

    session.add(
        AvailabilityOverride(
            start_time=block_start,
            end_time=block_end,
            reason="PTO",
        )
    )
    session.commit()

    ok, reason = calendar.is_slot_available(
        session,
        settings=settings,
        start_time=_local_dt(settings, day, 10, 0),
        duration_minutes=30,
    )

    assert ok is False
    assert reason is not None
    assert "blocked" in reason.lower() or "unavailable" in reason.lower()


# --------------------------------------------------------------------------- #
# get_available_slots
# --------------------------------------------------------------------------- #
def test_get_available_slots_counts_full_working_day(session, settings: Settings) -> None:
    """A single empty working day yields the expected number of 30-min slots.

    Window is 09:00..17:00 (8 hours) stepped by 30 minutes; a 30-minute meeting
    must fully fit, so the last start is 16:30 → 16 slots.
    """

    day = _next_working_day(settings)
    slots = calendar.get_available_slots(
        session, settings=settings, date_from=day, date_to=day, duration_minutes=30
    )

    assert len(slots) == 16
    # Slots are chronologically ordered, in the business timezone.
    tz = ZoneInfo(settings.timezone)
    assert slots[0].start == _local_dt(settings, day, 9, 0)
    assert slots[-1].start == _local_dt(settings, day, 16, 30)
    assert all(s.start.tzinfo is not None for s in slots)
    assert all(str(s.start.tzinfo) == str(tz) for s in slots)
    # Each slot is exactly the requested duration.
    assert all((s.end - s.start) == timedelta(minutes=30) for s in slots)


def test_get_available_slots_drops_overridden_day(session, settings: Settings) -> None:
    """A day fully covered by an override yields no slots."""

    day = _next_working_day(settings)
    block_start = _local_dt(settings, day, 9, 0).astimezone(timezone.utc)
    block_end = _local_dt(settings, day, 17, 0).astimezone(timezone.utc)
    session.add(
        AvailabilityOverride(start_time=block_start, end_time=block_end, reason="Holiday")
    )
    session.commit()

    slots = calendar.get_available_slots(
        session, settings=settings, date_from=day, date_to=day, duration_minutes=30
    )

    assert slots == []


def test_get_available_slots_drops_booked_slot(session, settings: Settings) -> None:
    """A confirmed booking removes exactly the overlapping slot(s)."""

    day = _next_working_day(settings)
    booked_start = _local_dt(settings, day, 10, 0).astimezone(timezone.utc)
    session.add(
        Booking(
            name="Taken",
            email="taken@example.com",
            start_time=booked_start,
            end_time=booked_start + timedelta(minutes=30),
            status="confirmed",
            channel="api",
        )
    )
    session.commit()

    slots = calendar.get_available_slots(
        session, settings=settings, date_from=day, date_to=day, duration_minutes=30
    )

    starts = {s.start for s in slots}
    assert _local_dt(settings, day, 10, 0) not in starts
    # One fewer slot than the full empty day (16 - 1 = 15).
    assert len(slots) == 15


def test_get_available_slots_excludes_past_today(session, settings: Settings) -> None:
    """Slots earlier than 'now' are excluded for the current day."""

    tz = ZoneInfo(settings.timezone)
    now_local = datetime.now(tz)
    today = now_local.date()

    if today.weekday() not in set(settings.working_days):
        pytest.skip("today is not a working day; past-slot pruning not observable")

    slots = calendar.get_available_slots(
        session, settings=settings, date_from=today, date_to=today, duration_minutes=30
    )

    now_utc = datetime.now(timezone.utc)
    assert all(s.start.astimezone(timezone.utc) >= now_utc for s in slots)


def test_get_available_slots_empty_when_range_inverted(session, settings: Settings) -> None:
    """An inverted date range (date_to < date_from) yields no slots."""

    day = _next_working_day(settings)
    earlier = day - timedelta(days=3)
    slots = calendar.get_available_slots(
        session, settings=settings, date_from=day, date_to=earlier
    )
    assert slots == []


# --------------------------------------------------------------------------- #
# check_availability tool
# --------------------------------------------------------------------------- #
def test_check_availability_tool_shape_and_cap(session, settings: Settings) -> None:
    """The tool returns the documented payload shape and caps slots at ~12."""

    day = _next_working_day(settings)
    payload = check_availability(
        {"date_from": day.isoformat(), "date_to": day.isoformat(), "duration_minutes": 30},
        session=session,
        settings=settings,
    )

    assert payload["timezone"] == settings.timezone
    assert payload["duration_minutes"] == 30
    assert isinstance(payload["slots"], list)
    assert payload["count"] == len(payload["slots"])
    # The full day has 16 open slots but the tool caps the return at 12.
    assert payload["count"] == 12
    assert all({"start", "end"} <= set(slot) for slot in payload["slots"])


# --------------------------------------------------------------------------- #
# book_meeting tool
# --------------------------------------------------------------------------- #
def test_book_meeting_confirms_then_rejects_double_book(session, settings: Settings) -> None:
    """Booking a free slot confirms; re-booking the same slot is unavailable."""

    day = _next_working_day(settings)
    start_iso = _local_dt(settings, day, 13, 0).isoformat()

    first = book_meeting(
        {
            "name": "Alice Example",
            "email": "alice@example.com",
            "start_time": start_iso,
            "topic": "Intro chat",
        },
        session=session,
        settings=settings,
        channel="api",
    )

    assert first["status"] == "confirmed"
    assert first["booking_id"]
    assert first["name"] == "Alice Example"
    assert first["email"] == "alice@example.com"
    assert first["topic"] == "Intro chat"
    assert first["timezone"] == settings.timezone
    # The persisted booking really exists and is confirmed.
    assert session.query(Booking).filter(Booking.id == first["booking_id"]).count() == 1

    second = book_meeting(
        {
            "name": "Bob Example",
            "email": "bob@example.com",
            "start_time": start_iso,
        },
        session=session,
        settings=settings,
        channel="api",
    )

    assert second["status"] == "unavailable"
    assert second["reason"]
    assert "booking" in second["reason"].lower()
    assert isinstance(second["alternatives"], list)
    assert 0 < len(second["alternatives"]) <= 3
    assert all({"start", "end"} <= set(alt) for alt in second["alternatives"])
    # No second booking row was created.
    assert session.query(Booking).filter(Booking.status == "confirmed").count() == 1


def test_book_meeting_rejects_bad_email(session, settings: Settings) -> None:
    """A malformed email raises before any booking is persisted."""

    day = _next_working_day(settings)
    start_iso = _local_dt(settings, day, 14, 0).isoformat()

    with pytest.raises(ValueError, match="not a valid email"):
        book_meeting(
            {
                "name": "No Email",
                "email": "not-an-email",
                "start_time": start_iso,
            },
            session=session,
            settings=settings,
            channel="api",
        )

    assert session.query(Booking).count() == 0


def test_book_meeting_requires_name(session, settings: Settings) -> None:
    """A missing/blank name raises a clear validation error."""

    day = _next_working_day(settings)
    start_iso = _local_dt(settings, day, 14, 0).isoformat()

    with pytest.raises(ValueError, match="name is required"):
        book_meeting(
            {"name": "   ", "email": "ok@example.com", "start_time": start_iso},
            session=session,
            settings=settings,
            channel="api",
        )


def test_book_meeting_unavailable_for_past_time(session, settings: Settings) -> None:
    """Booking a past time is reported as unavailable (not an exception)."""

    past_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    result = book_meeting(
        {"name": "Past Booker", "email": "past@example.com", "start_time": past_iso},
        session=session,
        settings=settings,
        channel="api",
    )

    assert result["status"] == "unavailable"
    assert "past" in result["reason"].lower()
    assert session.query(Booking).count() == 0
