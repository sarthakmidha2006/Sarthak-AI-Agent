"""Vapi telephony service — outbound call placement.

This is the single boundary to Vapi's REST API for *outbound* calls. Routes call
:func:`place_outbound_call`; no HTTP lives in the route layer. The function
returns a small, typed result so the route can map it to the public
``CallMeResponse`` without knowing anything about Vapi.

Vapi outbound call API (https://docs.vapi.ai/api-reference/calls/create):

    POST {base}/call
    Authorization: Bearer <VAPI_API_KEY>
    {
      "assistantId":   "<assistant id>",
      "phoneNumberId": "<vapi phone-number id to call FROM>",
      "customer": { "number": "+E164", "name": "..." }
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 15.0


@dataclass
class OutboundCallResult:
    """Outcome of an outbound-call attempt.

    Attributes:
        success: Whether Vapi accepted and queued the call.
        message: Human-readable status / error suitable for the API response.
        call_id: Vapi call id when the call was created, else ``None``.
    """

    success: bool
    message: str
    call_id: str | None = None


def _mask_phone(phone: str) -> str:
    """Mask a phone number for logs (keep country prefix + last 2 digits)."""
    if len(phone) <= 5:
        return "***"
    return f"{phone[:3]}…{phone[-2:]}"


def _missing_config(settings: Settings) -> list[str]:
    """Return the names of any required Vapi settings that are unset."""
    required = {
        "VAPI_API_KEY": settings.vapi_api_key,
        "VAPI_ASSISTANT_ID": settings.vapi_assistant_id,
        "VAPI_PHONE_NUMBER_ID": settings.vapi_phone_number_id,
    }
    return [name for name, value in required.items() if not value]


async def place_outbound_call(
    *, settings: Settings, phone: str, name: str | None = None
) -> OutboundCallResult:
    """Place an outbound call via Vapi, handled by the configured assistant.

    Args:
        settings: Application settings carrying Vapi credentials/ids.
        phone: Destination number in E.164 (validated by the caller/schema).
        name: Optional caller display name passed to the assistant as customer context.

    Returns:
        An :class:`OutboundCallResult`. Never raises for expected failures
        (misconfiguration, Vapi 4xx/5xx, network/timeout) — those are logged and
        returned as ``success=False`` with a useful message.
    """
    missing = _missing_config(settings)
    if missing:
        logger.error("Vapi outbound call not configured; missing: %s", ", ".join(missing))
        return OutboundCallResult(
            success=False,
            message="Calling service is not configured.",
        )

    payload = {
        "assistantId": settings.vapi_assistant_id,
        "phoneNumberId": settings.vapi_phone_number_id,
        "customer": {"number": phone, **({"name": name} if name else {})},
    }
    url = f"{settings.vapi_base_url.rstrip('/')}/call"
    headers = {
        "Authorization": f"Bearer {settings.vapi_api_key}",
        "Content-Type": "application/json",
    }

    log_ctx = {"phone": _mask_phone(phone), "assistant_id": settings.vapi_assistant_id}
    logger.info("Placing Vapi outbound call %s", log_ctx)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            res = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        logger.warning("Vapi outbound call timed out %s", log_ctx)
        return OutboundCallResult(success=False, message="The call service timed out. Please try again.")
    except httpx.HTTPError:
        logger.exception("Vapi outbound call transport error %s", log_ctx)
        return OutboundCallResult(
            success=False, message="Could not reach the call service. Please try again."
        )

    if res.status_code in (200, 201):
        body = _safe_json(res)
        call_id = body.get("id") if isinstance(body, dict) else None
        logger.info("Vapi outbound call queued id=%s %s", call_id, log_ctx)
        return OutboundCallResult(
            success=True, message="Call queued successfully.", call_id=call_id
        )

    # Non-2xx: surface a clean message, log the provider detail.
    detail = _error_detail(res)
    logger.error(
        "Vapi outbound call failed status=%s detail=%s %s", res.status_code, detail, log_ctx
    )
    if res.status_code in (401, 403):
        return OutboundCallResult(success=False, message="Calling service authentication failed.")
    if res.status_code == 400:
        return OutboundCallResult(
            success=False, message="The phone number was rejected by the call service."
        )
    return OutboundCallResult(success=False, message="Unable to place call. Please try again.")


def _safe_json(res: httpx.Response) -> object:
    try:
        return res.json()
    except ValueError:
        return None


def _error_detail(res: httpx.Response) -> str:
    """Extract a short error string from a Vapi error response for logging."""
    body = _safe_json(res)
    if isinstance(body, dict):
        msg = body.get("message") or body.get("error") or body.get("detail")
        if isinstance(msg, list):
            return "; ".join(str(m) for m in msg)[:300]
        if msg:
            return str(msg)[:300]
    return (res.text or "")[:300]
