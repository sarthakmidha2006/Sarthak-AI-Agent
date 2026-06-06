"""Call-Me route -- ``POST /call-me`` (outbound AI callback).

Accepts a visitor's name + E.164 phone number and asks Vapi to place an
outbound call handled by the existing assistant. All telephony/HTTP logic lives
in :mod:`app.services.vapi`; this route only validates input (via the
:class:`CallMeRequest` schema), invokes the service, and maps the result onto
the public :class:`CallMeResponse`.

The frontend's `lib/call-service.ts` → `/api/call-me` (Next route) forwards here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_settings_dep
from app.config import Settings
from app.models.api_schemas import CallMeRequest, CallMeResponse
from app.services.vapi import place_outbound_call

logger = logging.getLogger(__name__)

router = APIRouter(tags=["call"])


@router.post("/call-me", response_model=CallMeResponse, summary="Request an outbound AI call")
async def call_me(
    payload: CallMeRequest,
    settings: Settings = Depends(get_settings_dep),
) -> CallMeResponse:
    """Place an outbound call to the visitor, handled by the persona's assistant.

    Returns ``{"success": true}`` once Vapi has queued the call. On any expected
    failure (misconfiguration, provider rejection, network/timeout) returns
    ``{"success": false, "message": "..."}`` with a user-safe message — the route
    never raises a 500 into the caller for these cases.
    """
    result = await place_outbound_call(
        settings=settings, phone=payload.phone, name=payload.name
    )

    if not result.success:
        logger.info("call-me request rejected: %s", result.message)

    return CallMeResponse(
        success=result.success,
        message=result.message,
        call_id=result.call_id,
    )
