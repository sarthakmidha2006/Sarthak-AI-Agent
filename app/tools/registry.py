"""Tool registry and dispatcher (BUILD_SPEC §7.1).

Exposes:

* :data:`TOOL_SCHEMAS` — the OpenAI ``tools`` array (``type: "function"``) for the
  two scheduling tools the persona may call: ``check_availability`` and
  ``book_meeting``. The descriptions explicitly instruct the model to call
  ``check_availability`` before booking, to never fabricate slots, and to echo
  times back exactly as returned.
* :func:`dispatch_tool` — the single entry point the brain loop uses to execute a
  tool by name. It routes to the concrete implementation, injects the shared
  ``session`` / ``settings`` / ``channel`` context, and captures **every**
  exception as ``{"error": ...}`` so a misbehaving tool can never crash the
  brain's tool loop.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.tools.availability import check_availability
from app.tools.booking import book_meeting

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "List open meeting time slots from the candidate's real calendar. "
                "ALWAYS call this before attempting to book a meeting so you can "
                "offer the user genuine, currently-available times. Never invent, "
                "guess, or assume slots — only the times returned by this tool are "
                "real. When you present slots to the user, echo the start and end "
                "times back exactly as returned, in the returned timezone. All "
                "arguments are optional; omit them to use sensible defaults "
                "(today through the booking horizon, default meeting duration)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": (
                            "Earliest date to search, formatted as 'YYYY-MM-DD'. "
                            "Defaults to today if omitted."
                        ),
                    },
                    "date_to": {
                        "type": "string",
                        "description": (
                            "Latest date to search, formatted as 'YYYY-MM-DD'. "
                            "Defaults to the configured booking horizon if omitted."
                        ),
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": (
                            "Desired meeting length in minutes. Defaults to the "
                            "standard meeting duration if omitted."
                        ),
                        "minimum": 1,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": (
                "Book a confirmed meeting at a specific start time. Only call this "
                "AFTER you have used check_availability and confirmed the chosen "
                "time is one of the slots that tool returned — never fabricate a "
                "time. The start_time must be an exact slot start in ISO-8601 "
                "format. If the slot turns out to be unavailable the tool returns "
                "alternative slots; offer those to the user and ask them to pick "
                "one. Echo all confirmed times back exactly as returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full name of the person booking the meeting.",
                    },
                    "email": {
                        "type": "string",
                        "description": "Contact email address for the attendee.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": (
                            "Meeting start time in ISO-8601 format (e.g. "
                            "'2026-06-10T14:00:00'). Must exactly match a slot "
                            "start returned by check_availability."
                        ),
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": (
                            "Meeting length in minutes. Defaults to the standard "
                            "meeting duration if omitted."
                        ),
                        "minimum": 1,
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional short description of the meeting topic.",
                    },
                },
                "required": ["name", "email", "start_time"],
                "additionalProperties": False,
            },
        },
    },
]


def dispatch_tool(
    name: str,
    arguments: dict,
    *,
    session: Session,
    settings: Settings,
    channel: str,
) -> dict:
    """Execute a registered tool by name and return its result dict.

    Routes ``check_availability`` and ``book_meeting`` to their implementations,
    injecting the shared ``session`` / ``settings`` (and ``channel`` for booking).
    An unknown tool name returns ``{"error": ...}``. Any exception raised by a
    tool — including argument-validation errors — is caught and returned as
    ``{"error": str(exc)}`` so the brain's tool loop never sees a raised
    exception.

    Parameters
    ----------
    name:
        The tool name requested by the model.
    arguments:
        The decoded JSON arguments object for the call (``{}`` if the model sent
        none).
    session:
        Active SQLAlchemy session for calendar/booking reads and writes.
    settings:
        Application settings.
    channel:
        Origin channel (``"chat"`` | ``"voice"`` | ``"api"``) propagated to
        ``book_meeting``.

    Returns
    -------
    dict
        The tool's result payload, or ``{"error": ...}`` on unknown tool /
        failure.
    """

    args: dict[str, Any] = arguments if isinstance(arguments, dict) else {}
    logger.debug("dispatch_tool: name=%s channel=%s args=%s", name, channel, args)

    try:
        if name == "check_availability":
            return check_availability(args, session=session, settings=settings)
        if name == "book_meeting":
            return book_meeting(args, session=session, settings=settings, channel=channel)
        logger.warning("dispatch_tool: unknown tool requested: %r", name)
        return {"error": f"Unknown tool: {name!r}"}
    except Exception as exc:  # noqa: BLE001 - intentional: never raise into brain loop
        logger.exception("dispatch_tool: tool %r failed", name)
        return {"error": str(exc)}
