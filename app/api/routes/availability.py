"""Availability route -- ``GET /availability``.

Exposes the same working-hours availability computation that the
``check_availability`` tool uses, but as a plain REST endpoint for clients that
want to render a calendar (spec section 14.2). It delegates to
:func:`app.scheduling.calendar.get_available_slots` and maps the resulting
:class:`~app.scheduling.calendar.Slot` objects onto an
:class:`~app.models.api_schemas.AvailabilityResponse`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_db, get_settings_dep
from app.config import Settings
from app.models.api_schemas import AvailabilityResponse, SlotView
from app.scheduling import calendar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scheduling"])


def _parse_date(value: str | None, *, field: str) -> date | None:
    """Parse a ``YYYY-MM-DD`` query parameter into a :class:`datetime.date`.

    Returns ``None`` when the value is absent so the calendar layer can apply its
    own defaults (today .. today + horizon).
    """

    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return date.fromisoformat(candidate)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid '{field}': expected ISO date 'YYYY-MM-DD', got {value!r}.",
        ) from exc


@router.get(
    "/availability",
    response_model=AvailabilityResponse,
    summary="List bookable time slots",
)
def availability(
    date_from: str | None = Query(
        default=None,
        description="Inclusive start date (YYYY-MM-DD). Defaults to today.",
    ),
    date_to: str | None = Query(
        default=None,
        description="Inclusive end date (YYYY-MM-DD). Defaults to today + horizon.",
    ),
    duration_minutes: int | None = Query(
        default=None,
        ge=1,
        le=24 * 60,
        description="Desired meeting length in minutes. Defaults to the configured default.",
    ),
    db: "Session" = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> AvailabilityResponse:
    """Return open meeting slots within the requested (or default) date window."""

    parsed_from = _parse_date(date_from, field="date_from")
    parsed_to = _parse_date(date_to, field="date_to")

    if parsed_from is not None and parsed_to is not None and parsed_to < parsed_from:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'date_to' must not be earlier than 'date_from'.",
        )

    try:
        slots = calendar.get_available_slots(
            db,
            settings=settings,
            date_from=parsed_from,
            date_to=parsed_to,
            duration_minutes=duration_minutes,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to a clean 500
        logger.exception("Failed to compute availability")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute availability.",
        ) from exc

    effective_duration = duration_minutes or settings.booking_default_duration
    slot_views = [SlotView(**slot.to_dict()) for slot in slots]

    return AvailabilityResponse(
        timezone=settings.timezone,
        duration_minutes=effective_duration,
        slots=slot_views,
        count=len(slot_views),
    )
