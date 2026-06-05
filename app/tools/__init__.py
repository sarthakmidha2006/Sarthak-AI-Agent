"""Tool-calling package for the AI Persona brain.

This package exposes the two scheduling tools the persona can invoke during a
chat/voice turn:

* :func:`app.tools.availability.check_availability` — list open meeting slots.
* :func:`app.tools.booking.book_meeting` — book a confirmed meeting.

The OpenAI-format tool schemas and the central :func:`dispatch_tool` router live
in :mod:`app.tools.registry`. The brain loop only ever calls ``dispatch_tool``
(never the tool functions directly) so that every tool invocation gets uniform
argument parsing, channel propagation, and exception capture.
"""

from __future__ import annotations

from app.tools.availability import check_availability
from app.tools.booking import book_meeting
from app.tools.registry import TOOL_SCHEMAS, dispatch_tool

__all__ = [
    "TOOL_SCHEMAS",
    "dispatch_tool",
    "check_availability",
    "book_meeting",
]
