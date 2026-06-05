"""Idempotent database seed helpers (spec §5.3).

Currently provides :func:`seed_availability_overrides`, which inserts a small
set of demonstration blocked windows the first time the system runs against an
empty :class:`~app.db.models.AvailabilityOverride` table. It is a safe no-op on
subsequent runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AvailabilityOverride

logger = logging.getLogger(__name__)


def seed_availability_overrides(session: Session) -> None:
    """Insert demo blocked windows if (and only if) the table is empty.

    This keeps the operation idempotent: once any override exists (seeded or
    operator-created) the function does nothing. The demo windows are anchored
    relative to "now" so they remain in the future regardless of when seeding
    runs.

    Args:
        session: An active SQLAlchemy session. The caller owns its lifecycle;
            this function commits its own inserts on success.
    """

    existing = session.scalar(select(AvailabilityOverride.id).limit(1))
    if existing is not None:
        logger.debug("Availability overrides already present; skipping seed")
        return

    now = datetime.now(timezone.utc)

    # A two-hour "team offsite" block tomorrow afternoon and a full-day
    # "holiday" block one week out. Both are illustrative defaults that an
    # operator can delete/replace freely.
    offsite_start = (now + timedelta(days=1)).replace(
        hour=21, minute=0, second=0, microsecond=0
    )
    offsite_end = offsite_start + timedelta(hours=2)

    holiday_start = (now + timedelta(days=7)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    holiday_end = holiday_start + timedelta(days=1)

    overrides = [
        AvailabilityOverride(
            start_time=offsite_start,
            end_time=offsite_end,
            reason="Demo: team offsite",
        ),
        AvailabilityOverride(
            start_time=holiday_start,
            end_time=holiday_end,
            reason="Demo: company holiday",
        ),
    ]

    session.add_all(overrides)
    session.commit()
    logger.info("Seeded %d demo availability override(s)", len(overrides))
