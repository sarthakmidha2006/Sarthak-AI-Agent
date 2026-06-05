"""The persona "brain" — the shared, channel-agnostic answering engine.

:class:`PersonaBrain` orchestrates a single answer end-to-end:

1. Scan the query with the prompt-injection guard; refuse (short-circuit) on a
   high-confidence attack.
2. Retrieve grounding context via the hybrid retriever.
3. Build the prompt and run the OpenAI tool-calling loop (``check_availability``
   / ``book_meeting``) up to ``settings.max_tool_iterations`` iterations.
4. Optionally verify the answer against the context with the grounding judge
   (skipped for tool-confirmation answers and when a booking changed state).
5. Build citations, persist the conversation / messages / query log, and return
   a fully populated :class:`BrainResponse` with latency tracking.

Both the chat and voice routes call :meth:`PersonaBrain.answer` — the brain is
deliberately channel-agnostic. Errors inside the tool loop are caught and
surfaced as a graceful assistant message; the loop never throws to the route.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from openai import RateLimitError

from app.brain.prompts import build_messages, citations_from_chunks
from app.security.grounding import check_grounding, is_tool_confirmation
from app.security.prompt_guard import REFUSAL_MESSAGE
from app.tools.registry import TOOL_SCHEMAS, dispatch_tool

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.brain.llm import LLMClient
    from app.config import Settings
    from app.rag.retriever import HybridRetriever
    from app.rag.schemas import RetrievalResult
    from app.security.prompt_guard import PromptGuard

logger = logging.getLogger(__name__)

# Friendly fallback used when the tool loop hits an unrecoverable error.
_TOOL_LOOP_ERROR_MESSAGE: str = (
    "I'm sorry — I ran into a problem while working on that. Please try again, "
    "and if you were trying to schedule a meeting, let me know the day and time "
    "that works for you."
)

# Fallback when the model produces no textual content at all.
_EMPTY_ANSWER_MESSAGE: str = (
    "I'm sorry, I wasn't able to produce a response to that. Could you rephrase "
    "your question?"
)

# Returned when the Groq backend reports a 429 (rate limit) that survives the
# client's retry/backoff budget. Kept distinct from _TOOL_LOOP_ERROR_MESSAGE so
# callers/users can tell a transient quota issue from a real failure.
_RATE_LIMIT_MESSAGE: str = (
    "The AI service is temporarily rate limited. Please try again shortly."
)


@dataclass
class BrainResponse:
    """Everything a route needs to render a chat or voice reply.

    Attributes mirror BUILD SPEC §12 exactly so the API layer can map this onto
    its pydantic response models without translation surprises.
    """

    answer: str
    citations: list[dict]
    tool_calls: list[dict]
    retrieval: RetrievalResult | None
    injection_flagged: bool
    grounded: bool | None
    conversation_id: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    latency_breakdown: dict = field(default_factory=dict)


class PersonaBrain:
    """Shared answering engine for chat and voice channels."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLMClient,
        guard: PromptGuard,
        settings: Settings,
        session_factory: Callable[[], Session],
    ) -> None:
        """Build the brain.

        Args:
            retriever: Hybrid (vector + BM25 + RRF + rerank) retriever.
            llm: Async OpenAI wrapper used for chat (and the grounding judge).
            guard: Prompt-injection guard.
            settings: Application settings.
            session_factory: Zero-arg callable returning a new SQLAlchemy
                ``Session`` (e.g. ``SessionLocal``). The brain owns the session
                lifecycle for the duration of one ``answer`` call.
        """
        self._retriever = retriever
        self._llm = llm
        self._guard = guard
        self._settings = settings
        self._session_factory = session_factory

    async def answer(
        self,
        query: str,
        *,
        channel: str,
        history: list[dict] | None = None,
        conversation_id: str | None = None,
    ) -> BrainResponse:
        """Answer ``query`` for the given ``channel``.

        See the module docstring / BUILD SPEC §12 for the full lifecycle. The
        method always closes its DB session and never raises out of the tool
        loop.

        Args:
            query: The user's question / utterance text.
            channel: ``"chat"`` | ``"voice"`` | ``"api"`` (affects bookings &
                prompt phrasing).
            history: Prior conversation turns (user/assistant), newest last.
            conversation_id: Existing conversation id to append to, or ``None``
                to start a new one.

        Returns:
            A fully populated :class:`BrainResponse`.
        """
        start = time.perf_counter()
        breakdown: dict[str, float] = {}
        session = self._session_factory()
        new_conversation = conversation_id is None
        conv_id = conversation_id or uuid4().hex

        try:
            # 1) Prompt-injection guard (short-circuit on high-confidence attack).
            guard_start = time.perf_counter()
            guard_result = await self._guard.scan(query)
            breakdown["guard"] = _elapsed_ms(guard_start)

            if guard_result.action == "refuse":
                logger.warning(
                    "Refusing query due to injection guard: reason=%s matched=%s",
                    guard_result.reason,
                    guard_result.matched_patterns,
                )
                latency_ms = _elapsed_ms(start)
                breakdown["total"] = latency_ms
                self._persist(
                    session=session,
                    new_conversation=new_conversation,
                    conv_id=conv_id,
                    channel=channel,
                    query=query,
                    answer=REFUSAL_MESSAGE,
                    retrieval=None,
                    citations=[],
                    tool_calls=[],
                    prompt_tokens=None,
                    completion_tokens=None,
                    latency_ms=latency_ms,
                    breakdown=breakdown,
                    injection_flagged=True,
                    grounded=None,
                )
                return BrainResponse(
                    answer=REFUSAL_MESSAGE,
                    citations=[],
                    tool_calls=[],
                    retrieval=None,
                    injection_flagged=True,
                    grounded=None,
                    conversation_id=conv_id,
                    prompt_tokens=None,
                    completion_tokens=None,
                    latency_ms=latency_ms,
                    latency_breakdown=breakdown,
                )

            injection_flagged = guard_result.flagged

            # 2) Retrieve grounding context.
            retrieval = await self._retrieve(query, breakdown)

            # 3) Build prompt + run tool loop.
            messages = build_messages(
                query=query,
                retrieval=retrieval,
                history=history,
                settings=self._settings,
                channel=channel,
            )
            loop_start = time.perf_counter()
            (
                answer_text,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                booking_changed_state,
            ) = await self._run_tool_loop(messages, channel=channel, session=session)
            breakdown["llm_loop"] = _elapsed_ms(loop_start)

            # 4) Grounding check (skip for tool confirmations / state changes).
            grounded = await self._maybe_check_grounding(
                answer_text,
                retrieval,
                tool_calls=tool_calls,
                booking_changed_state=booking_changed_state,
                breakdown=breakdown,
            )

            # 5) Citations.
            citations = citations_from_chunks(retrieval.chunks)

            latency_ms = _elapsed_ms(start)
            breakdown["total"] = latency_ms

            # 6) Persist conversation, messages, and the query log.
            self._persist(
                session=session,
                new_conversation=new_conversation,
                conv_id=conv_id,
                channel=channel,
                query=query,
                answer=answer_text,
                retrieval=retrieval,
                citations=citations,
                tool_calls=tool_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                breakdown=breakdown,
                injection_flagged=injection_flagged,
                grounded=grounded,
            )

            return BrainResponse(
                answer=answer_text,
                citations=citations,
                tool_calls=tool_calls,
                retrieval=retrieval,
                injection_flagged=injection_flagged,
                grounded=grounded,
                conversation_id=conv_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                latency_breakdown=breakdown,
            )
        finally:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - never let cleanup mask the result
                logger.exception("Failed to close DB session after answer()")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _retrieve(self, query: str, breakdown: dict[str, float]) -> RetrievalResult:
        """Run retrieval, recording its timing; degrade to an empty result.

        Retrieval failures must not abort the answer — the persona simply has no
        context and will say it doesn't have the information.
        """
        retrieve_start = time.perf_counter()
        try:
            retrieval = await self._retriever.retrieve(query)
        except Exception:  # noqa: BLE001 - degrade gracefully on retrieval failure
            logger.exception("Retrieval failed; proceeding with empty context")
            retrieval = self._empty_retrieval(query)
        breakdown["retrieval"] = _elapsed_ms(retrieve_start)
        return retrieval

    @staticmethod
    def _empty_retrieval(query: str) -> RetrievalResult:
        """Build an empty :class:`RetrievalResult` for degraded retrieval."""
        from app.rag.schemas import RetrievalResult as _RetrievalResult

        return _RetrievalResult(
            query=query,
            chunks=[],
            timings_ms={"total": 0.0},
            candidate_count=0,
        )

    async def _run_tool_loop(
        self,
        messages: list[dict],
        *,
        channel: str,
        session: Session,
    ) -> tuple[str, list[dict], int | None, int | None, bool]:
        """Run the chat + tool-calling loop.

        Returns ``(answer_text, tool_calls, prompt_tokens, completion_tokens,
        booking_changed_state)``. ``tool_calls`` is the list of
        ``{"name", "arguments", "result"}`` records (one per executed call).
        ``booking_changed_state`` is ``True`` when a ``book_meeting`` call
        returned a confirmed booking.

        All exceptions are caught and surfaced as a graceful assistant message;
        the loop never raises to the caller.
        """
        tool_calls: list[dict] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        answer_text = ""
        booking_changed_state = False
        max_iters = max(1, int(self._settings.max_tool_iterations))
        # Signature of the tool call(s) executed on the previous iteration, used
        # to detect a model that re-requests the same tool with the same
        # arguments on consecutive turns (see the repeated-call guard below).
        last_tool_signature: tuple | None = None

        for iteration in range(max_iters):
            try:
                result = await self._llm.chat(
                    messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self._settings.chat_temperature,
                )
            except RateLimitError:
                # Explicit 429: tell the user the service is rate-limited rather
                # than emitting a generic failure. Citations are attached by the
                # caller from the already-completed retrieval, so the response
                # still carries useful sources.
                logger.warning("Groq rate limit (429) during tool loop (iter=%d)", iteration)
                return (
                    _RATE_LIMIT_MESSAGE,
                    tool_calls,
                    prompt_tokens,
                    completion_tokens,
                    booking_changed_state,
                )
            except Exception:  # noqa: BLE001 - surface as graceful message
                logger.exception("LLM chat call failed during tool loop (iter=%d)", iteration)
                return (
                    _TOOL_LOOP_ERROR_MESSAGE,
                    tool_calls,
                    prompt_tokens,
                    completion_tokens,
                    booking_changed_state,
                )

            prompt_tokens = _accumulate_tokens(prompt_tokens, result.prompt_tokens)
            completion_tokens = _accumulate_tokens(completion_tokens, result.completion_tokens)

            assistant_msg = result.message or {}
            raw_tool_calls = assistant_msg.get("tool_calls") or []

            if not raw_tool_calls:
                # Final answer.
                content = assistant_msg.get("content")
                answer_text = content if isinstance(content, str) else ""
                break

            # Guard against a model that re-requests the SAME tool with the SAME
            # arguments on consecutive iterations (observed with Groq llama-3.3
            # under tool_choice="auto"). The previous round's identical tool
            # result is already in `messages`, so stop looping and force a
            # natural-language answer from that most recent result rather than
            # executing the duplicate call again.
            current_signature = self._tool_call_signature(raw_tool_calls)
            if last_tool_signature is not None and current_signature == last_tool_signature:
                logger.warning(
                    "Repeated identical tool call(s) detected (%s); stopping tool "
                    "loop and forcing a final answer from the latest tool result",
                    current_signature,
                )
                answer_text = await self._force_final_answer(messages)
                break
            last_tool_signature = current_signature

            # The model wants to call tools. Ensure every call has a stable id
            # ONCE, before it is consumed by both the assistant message and its
            # matching tool-result message — otherwise a missing id would be
            # filled with two different uuids and the tool_call_id pairing the
            # OpenAI API requires would break. (The SDK normally supplies ids;
            # this guards the rare missing-id case deterministically.)
            for _call in raw_tool_calls:
                if isinstance(_call, dict) and not _call.get("id"):
                    _call["id"] = uuid4().hex

            # Append the assistant message (with tool_calls) verbatim, then
            # execute each call and append a tool result keyed by the call id.
            messages.append(self._assistant_tool_message(assistant_msg, raw_tool_calls))

            for call in raw_tool_calls:
                name, arguments, tool_result, call_id = self._execute_tool_call(
                    call, channel=channel, session=session
                )
                tool_calls.append(
                    {"name": name, "arguments": arguments, "result": tool_result}
                )
                if name == "book_meeting" and isinstance(tool_result, dict):
                    if tool_result.get("status") == "confirmed":
                        booking_changed_state = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(tool_result, default=str),
                    }
                )
            # Continue the loop so the model can incorporate tool results.
        else:
            # Exhausted iterations without a final textual answer. Ask once more
            # without tools to force a natural-language wrap-up.
            logger.warning("Tool loop exhausted %d iterations; forcing final answer", max_iters)
            answer_text = await self._force_final_answer(messages)

        if not answer_text or not answer_text.strip():
            # The model returned no content (e.g. only tool calls were produced).
            if booking_changed_state or tool_calls:
                answer_text = await self._force_final_answer(messages)
            if not answer_text or not answer_text.strip():
                answer_text = _EMPTY_ANSWER_MESSAGE

        return (
            answer_text,
            tool_calls,
            prompt_tokens,
            completion_tokens,
            booking_changed_state,
        )

    async def _force_final_answer(self, messages: list[dict]) -> str:
        """Request a final natural-language answer with tools disabled."""
        try:
            result = await self._llm.chat(
                messages,
                tools=None,
                tool_choice="none",
                temperature=self._settings.chat_temperature,
            )
        except RateLimitError:
            logger.warning("Groq rate limit (429) while forcing final answer")
            return _RATE_LIMIT_MESSAGE
        except Exception:  # noqa: BLE001 - graceful fallback
            logger.exception("Failed to force final answer after tool loop")
            return _TOOL_LOOP_ERROR_MESSAGE
        content = (result.message or {}).get("content")
        return content if isinstance(content, str) and content.strip() else ""

    @staticmethod
    def _tool_call_signature(raw_tool_calls: list[dict]) -> tuple:
        """Build a hashable signature of a round's tool calls (name + arguments).

        Arguments are canonicalized — parsed JSON re-dumped with sorted keys —
        so that semantically identical calls compare equal regardless of key
        ordering or whitespace. Used to detect a model that re-requests the same
        tool with the same arguments on consecutive iterations.

        Args:
            raw_tool_calls: The ``tool_calls`` array from an assistant message.

        Returns:
            A tuple of ``(name, canonical_arguments)`` pairs, one per call.
        """
        signature: list[tuple[str, str]] = []
        for call in raw_tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name", "") or "unknown"
            raw_arguments = function.get("arguments", "{}")
            if isinstance(raw_arguments, dict):
                args_obj: object = raw_arguments
            else:
                try:
                    args_obj = json.loads(raw_arguments or "{}")
                except (ValueError, TypeError):
                    args_obj = None
            if isinstance(args_obj, dict):
                canonical = json.dumps(args_obj, sort_keys=True, default=str)
            else:
                canonical = str(raw_arguments)
            signature.append((name, canonical))
        return tuple(signature)

    @staticmethod
    def _assistant_tool_message(assistant_msg: dict, raw_tool_calls: list[dict]) -> dict:
        """Normalize the assistant tool-call message for the message history.

        Ensures the structure the API expects: ``role='assistant'``, optional
        ``content``, and a ``tool_calls`` array with ``id`` /
        ``function.name`` / ``function.arguments``.
        """
        normalized_calls: list[dict] = []
        for call in raw_tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            normalized_calls.append(
                {
                    "id": (call.get("id") if isinstance(call, dict) else None) or uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    },
                }
            )
        return {
            "role": "assistant",
            "content": assistant_msg.get("content") or "",
            "tool_calls": normalized_calls,
        }

    def _execute_tool_call(
        self,
        call: dict,
        *,
        channel: str,
        session: Session,
    ) -> tuple[str, dict, dict, str]:
        """Parse and dispatch a single tool call.

        Returns ``(name, parsed_arguments, result_dict, call_id)``. Argument
        JSON-parse errors and dispatch errors are caught and turned into an
        ``{"error": ...}`` result so the loop continues gracefully.
        """
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name", "") or "unknown"
        call_id = (call.get("id") if isinstance(call, dict) else None) or uuid4().hex
        raw_arguments = function.get("arguments", "{}")

        arguments: dict
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                parsed = json.loads(raw_arguments or "{}")
                arguments = parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                logger.warning(
                    "Tool '%s' had non-JSON arguments: %r", name, raw_arguments
                )
                return (
                    name,
                    {},
                    {"error": "Could not parse tool arguments as JSON."},
                    call_id,
                )

        try:
            result = dispatch_tool(
                name,
                arguments,
                session=session,
                settings=self._settings,
                channel=channel,
            )
            if not isinstance(result, dict):
                result = {"result": result}
        except Exception as exc:  # noqa: BLE001 - dispatch must not break the loop
            logger.exception("dispatch_tool('%s') raised", name)
            result = {"error": str(exc)}

        return name, arguments, result, call_id

    async def _maybe_check_grounding(
        self,
        answer_text: str,
        retrieval: RetrievalResult,
        *,
        tool_calls: list[dict],
        booking_changed_state: bool,
        breakdown: dict[str, float],
    ) -> bool | None:
        """Run the grounding judge when appropriate.

        Skipped (returns ``None``) when grounding is disabled, when a booking
        changed state, or when the answer is a tool-confirmation. Otherwise runs
        the judge and records its timing.
        """
        if not self._settings.grounding_check_enabled:
            return None
        if booking_changed_state:
            return None
        # Only treat the answer as a tool-confirmation (and skip grounding) when a
        # tool actually ran this turn. Otherwise an ordinary corpus answer that
        # merely uses words like "scheduled"/"availability" would dodge the
        # grounding judge — exactly the hallucinations we want to catch.
        if tool_calls and is_tool_confirmation(answer_text):
            return None
        # Tools may have run without a booking state change (e.g. an availability
        # lookup) yet still produce corpus-grounded prose; in that case fall
        # through and judge it.

        ground_start = time.perf_counter()
        try:
            result = await check_grounding(
                answer_text, retrieval.chunks, self._llm, self._settings
            )
        except Exception:  # noqa: BLE001 - fail open
            logger.exception("Grounding check raised; failing open (grounded=True)")
            breakdown["grounding"] = _elapsed_ms(ground_start)
            return True
        breakdown["grounding"] = _elapsed_ms(ground_start)
        if not result.grounded:
            logger.warning(
                "Answer flagged as not fully grounded (score=%.2f, unsupported=%s)",
                result.score,
                result.unsupported_claims,
            )
        return result.grounded

    def _persist(
        self,
        *,
        session: Session,
        new_conversation: bool,
        conv_id: str,
        channel: str,
        query: str,
        answer: str,
        retrieval: RetrievalResult | None,
        citations: list[dict],
        tool_calls: list[dict],
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: float,
        breakdown: dict,
        injection_flagged: bool,
        grounded: bool | None,
    ) -> None:
        """Persist the conversation, user/assistant messages, and query log.

        Persistence failures are logged and swallowed: a logging/DB hiccup must
        never prevent the user from receiving their answer. The session is rolled
        back on failure so the surrounding ``finally`` can close it cleanly.
        """
        from app.db.models import Conversation, Message, QueryLog

        try:
            now = datetime.now(timezone.utc)

            if new_conversation:
                session.add(
                    Conversation(id=conv_id, channel=channel, created_at=now)
                )

            session.add(
                Message(
                    id=uuid4().hex,
                    conversation_id=conv_id,
                    role="user",
                    content=query,
                    created_at=now,
                )
            )
            session.add(
                Message(
                    id=uuid4().hex,
                    conversation_id=conv_id,
                    role="assistant",
                    content=answer,
                    created_at=datetime.now(timezone.utc),
                )
            )

            retrieved_chunk_ids = (
                [scored.chunk.id for scored in retrieval.chunks] if retrieval else []
            )
            session.add(
                QueryLog(
                    id=uuid4().hex,
                    conversation_id=conv_id,
                    channel=channel,
                    query=query,
                    answer=answer,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    citations=citations,
                    tool_calls=tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms_total=latency_ms,
                    latency_breakdown=breakdown,
                    injection_flagged=injection_flagged,
                    grounded=grounded,
                    created_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
        except Exception:  # noqa: BLE001 - persistence must not break the response
            logger.exception("Failed to persist conversation/query log; rolling back")
            try:
                session.rollback()
            except Exception:  # noqa: BLE001
                logger.exception("Rollback also failed after persistence error")


def _elapsed_ms(since: float) -> float:
    """Return milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return round((time.perf_counter() - since) * 1000.0, 3)


def _accumulate_tokens(current: int | None, addition: int | None) -> int | None:
    """Sum token counts across tool-loop iterations, tolerating ``None``."""
    if addition is None:
        return current
    if current is None:
        return addition
    return current + addition
