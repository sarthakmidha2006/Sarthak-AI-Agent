"""Booking route -- ``POST /book``.

Books a meeting directly over REST, bypassing the LLM tool loop but reusing the
*exact same* :func:`app.tools.booking.book_meeting` implementation that the brain
invokes (spec sections 7.3 and 14.2). The shared implementation guarantees that
REST and tool-driven bookings apply identical validation (email format, working
hours, double-booking / override conflicts) and persist identically.

The channel is fixed to ``"api"`` so bookings made via this route are
distinguishable from ``"chat"`` / ``"voice"`` bookings in the database.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_db, get_settings_dep
from app.config import Settings
from app.models.api_schemas import BookRequest, BookResponse, SlotView
from app.tools import booking as booking_tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scheduling"])


def _alternatives_to_views(raw: Any) -> list[SlotView] | None:
    """Coerce the tool's ``alternatives`` payload into typed :class:`SlotView`s."""

    if not raw:
        return None
    views: list[SlotView] = []
    for item in raw:
        if isinstance(item, dict) and "start" in item and "end" in item:
            views.append(SlotView(start=str(item["start"]), end=str(item["end"])))
    return views or None


def _to_book_response(result: dict[str, Any]) -> BookResponse:
    """Map the :func:`book_meeting` result dict onto a :class:`BookResponse`.

    ``book_meeting`` returns one of two shapes:

    * ``{"status": "confirmed", "booking_id", "start_time", "end_time", ...}``
    * ``{"status": "unavailable", "reason", "alternatives": [...]}``

    We synthesise a human-readable ``message`` for both and surface alternatives
    when the requested slot was unavailable.
    """

    raw_status = str(result.get("status", "error"))

    if raw_status == "confirmed":
        message = "Your meeting has been confirmed."
        return BookResponse(
            status=raw_status,
            message=message,
            booking_id=result.get("booking_id"),
            start_time=result.get("start_time"),
            end_time=result.get("end_time"),
            timezone=result.get("timezone"),
            alternatives=None,
        )

    if raw_status == "unavailable":
        reason = result.get("reason") or "The requested time is not available."
        return BookResponse(
            status=raw_status,
            message=str(reason),
            booking_id=None,
            start_time=None,
            end_time=None,
            timezone=result.get("timezone"),
            alternatives=_alternatives_to_views(result.get("alternatives")),
        )

    # Defensive fallback: the tool reported an error (e.g. invalid input it
    # validated internally). Surface it without leaking internals.
    message = str(result.get("error") or result.get("reason") or "Booking failed.")
    return BookResponse(
        status="error",
        message=message,
        booking_id=None,
        start_time=None,
        end_time=None,
        timezone=result.get("timezone"),
        alternatives=None,
    )


@router.post("/book", response_model=BookResponse, summary="Book a meeting")
def book(
    payload: BookRequest,
    db: "Session" = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> BookResponse:
    """Validate and persist a booking via the shared ``book_meeting`` tool."""

    arguments: dict[str, Any] = {
        "name": payload.name,
        "email": str(payload.email),
        "start_time": payload.start_time,
    }
    if payload.duration_minutes is not None:
        arguments["duration_minutes"] = payload.duration_minutes
    if payload.topic is not None:
        arguments["topic"] = payload.topic

    try:
        result = booking_tool.book_meeting(
            arguments,
            session=db,
            settings=settings,
            channel="api",
        )
    except Exception as exc:  # noqa: BLE001 - degrade to a clean 500
        logger.exception("Booking failed for %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process booking.",
        ) from exc

    return _to_book_response(result)
