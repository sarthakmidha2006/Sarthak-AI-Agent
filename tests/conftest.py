"""Shared pytest fixtures and offline test doubles (spec §17).

This module wires the whole suite for **fully offline** execution:

* A throwaway, temp-file SQLite database is created once per test session and
  the schema is initialised with :func:`app.db.database.init_db`. The module
  level :data:`app.db.database.engine` / :data:`app.db.database.SessionLocal`
  are rebound to the test engine so *every* code path that reaches the database
  through those globals (the scheduling calendar, the ``book_meeting`` tool, and
  the ``get_db`` FastAPI dependency) transparently uses the test database.
* :class:`FakeLLM` stands in for :class:`app.brain.llm.LLMClient`. It returns
  *scripted* chat / tool-call responses and *deterministic* embeddings, so tests
  never touch the network and never require an OpenAI API key.
* Convenience fixtures expose a :class:`~app.config.Settings` instance and a
  ready-to-use SQLAlchemy :class:`~sqlalchemy.orm.Session`.

The fixtures here are deliberately conservative: nothing imports
``chromadb``/``openai`` network clients at collection time, keeping the suite
import-safe in minimal environments.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings


# --------------------------------------------------------------------------- #
# Database: a temp-file SQLite engine wired into app.db.database globals.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def _db_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    """Create a temp-file SQLite engine and rebind the app's DB globals to it.

    A temp *file* (rather than ``:memory:``) is used so the single engine can be
    shared across every connection/thread for the whole session without the
    in-memory-per-connection isolation surprise. The original module globals are
    restored on teardown so importing the package elsewhere is unaffected.
    """

    from app.db import database as db_module

    db_path = tmp_path_factory.mktemp("persona-db") / "test_persona.db"
    test_url = f"sqlite:///{db_path}"

    test_engine = db_module._build_engine(test_url)
    test_session_factory = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
        future=True,
    )

    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal
    original_settings_url = db_module._settings.database_url

    # Rebind every reference the rest of the codebase resolves at call time.
    db_module.engine = test_engine
    db_module.SessionLocal = test_session_factory
    db_module._settings.database_url = test_url

    # Create the schema against the test engine.
    db_module.Base.metadata.create_all(bind=test_engine)
    # init_db() is idempotent and exercises the real code path too.
    db_module.init_db()

    try:
        yield test_engine
    finally:
        db_module.Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session_local
        db_module._settings.database_url = original_settings_url


@pytest.fixture()
def session_factory(_db_engine: Engine) -> Callable[[], Session]:
    """Return a zero-arg callable producing fresh sessions bound to the test DB."""

    return sessionmaker(
        bind=_db_engine,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
        future=True,
    )


@pytest.fixture()
def session(session_factory: Callable[[], Session]) -> Iterator[Session]:
    """Yield a single SQLAlchemy session bound to the (clean) test database.

    Every booking / override row created during a test is removed afterwards so
    tests remain order-independent without paying for a full schema rebuild.
    """

    from app.db.models import (
        AvailabilityOverride,
        Booking,
        Conversation,
        EvalResult,
        Message,
        QueryLog,
    )

    db = session_factory()
    try:
        yield db
    finally:
        db.rollback()
        # Best-effort cleanup so tables are empty for the next test.
        for model in (
            QueryLog,
            Message,
            Conversation,
            Booking,
            AvailabilityOverride,
            EvalResult,
        ):
            try:
                db.query(model).delete()
            except Exception:  # noqa: BLE001 - cleanup must not mask test failures
                db.rollback()
        db.commit()
        db.close()


# --------------------------------------------------------------------------- #
# Settings fixture.
# --------------------------------------------------------------------------- #
@pytest.fixture()
def settings() -> Settings:
    """Return deterministic :class:`~app.config.Settings` for tests.

    The values are pinned (rather than read from the environment) so scheduling
    math, chunking sizes, and security toggles are stable regardless of the host
    ``.env``. The grounding/injection LLM features are kept off by default; the
    individual security tests that exercise them flip the flags explicitly.
    """

    return Settings(
        openai_api_key="test-key-not-used",
        timezone="America/Los_Angeles",
        working_days=[0, 1, 2, 3, 4],
        working_hours_start=9,
        working_hours_end=17,
        slot_minutes=30,
        booking_default_duration=30,
        booking_horizon_days=14,
        chunk_size_tokens=64,
        chunk_overlap_tokens=16,
        injection_guard_enabled=True,
        injection_llm_classifier=False,
        grounding_check_enabled=True,
        final_context_chunks=4,
        rerank_candidates=8,
        top_k_vector=8,
        top_k_bm25=8,
    )


# --------------------------------------------------------------------------- #
# FakeLLM: scripted, deterministic, network-free stand-in for LLMClient.
# --------------------------------------------------------------------------- #
def _deterministic_embedding(text: str, dim: int = 16) -> list[float]:
    """Return a stable, L2-normalised embedding derived only from ``text``.

    The vector is produced from a SHA-256 digest of the text so the same input
    always yields the same output (no randomness, no network). It is unit-length
    so cosine-similarity-based callers behave sensibly, but the values are
    otherwise meaningless — fine for tests that only need determinism/alignment.
    """

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


@dataclass
class _ScriptedChat:
    """One scripted assistant turn for :class:`FakeLLM`.

    Attributes:
        content: The assistant message text (``None`` when only tool calls are
            emitted).
        tool_calls: A list of ``(name, arguments_json)`` tuples. When non-empty
            the produced message carries an OpenAI-shaped ``tool_calls`` array
            and the finish reason is ``"tool_calls"``.
        finish_reason: Override for the finish reason; defaults are derived from
            whether tool calls are present.
    """

    content: str | None = None
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    finish_reason: str | None = None


class FakeLLM:
    """A scripted, deterministic stand-in for :class:`app.brain.llm.LLMClient`.

    The public surface mirrors :class:`LLMClient` (``chat`` / ``embed`` /
    ``transcribe`` / ``synthesize``) so it can be dropped into anything that
    depends on the real client without code changes.

    Chat behaviour:
        * If a list of scripted turns is supplied, each :meth:`chat` call pops
          the next turn (the last one repeats once the script is exhausted).
        * Otherwise a single default text turn is returned.

    Every call is recorded on :attr:`chat_calls` / :attr:`embed_calls` /
    :attr:`transcribe_calls` / :attr:`synthesize_calls` for assertions.
    """

    def __init__(
        self,
        *,
        scripted_chats: list[_ScriptedChat] | None = None,
        default_answer: str = "This is a grounded answer [1].",
        embedding_dim: int = 16,
        grounding_json: str | None = None,
        classifier_json: str | None = None,
    ) -> None:
        self._scripted_chats = list(scripted_chats or [])
        self._default_answer = default_answer
        self._embedding_dim = embedding_dim
        # Optional canned JSON for json_object responses (grounding / classifier).
        self._grounding_json = grounding_json
        self._classifier_json = classifier_json

        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []
        self.transcribe_calls: list[bytes] = []
        self.synthesize_calls: list[str] = []
        self._chat_index = 0

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> Any:
        """Return a scripted :class:`~app.brain.llm.ChatResult`-shaped object."""

        from app.brain.llm import ChatResult

        self.chat_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "response_format": response_format,
            }
        )

        # JSON-mode requests (grounding judge / injection classifier) get canned
        # JSON when provided, so those code paths are exercised deterministically.
        if response_format and response_format.get("type") == "json_object":
            canned = self._json_for_messages(messages)
            return ChatResult(
                message={"role": "assistant", "content": canned},
                finish_reason="stop",
                prompt_tokens=11,
                completion_tokens=7,
            )

        turn = self._next_turn()
        message = self._turn_to_message(turn)
        finish_reason = turn.finish_reason or ("tool_calls" if turn.tool_calls else "stop")
        return ChatResult(
            message=message,
            finish_reason=finish_reason,
            prompt_tokens=13,
            completion_tokens=5,
        )

    def _next_turn(self) -> _ScriptedChat:
        """Pop the next scripted turn, repeating the last once exhausted."""

        if not self._scripted_chats:
            return _ScriptedChat(content=self._default_answer)
        if self._chat_index < len(self._scripted_chats):
            turn = self._scripted_chats[self._chat_index]
            self._chat_index += 1
            return turn
        return self._scripted_chats[-1]

    def _json_for_messages(self, messages: list[dict]) -> str:
        """Pick the appropriate canned JSON for a json_object request."""

        system = ""
        for msg in messages:
            if msg.get("role") == "system":
                system = str(msg.get("content", "")).lower()
                break
        if "classifier" in system or "injection" in system:
            return self._classifier_json or '{"injection": false, "reason": "benign"}'
        # Default to the grounding judge shape.
        return self._grounding_json or '{"claims": [], "unsupported_claims": []}'

    @staticmethod
    def _turn_to_message(turn: _ScriptedChat) -> dict:
        """Convert a scripted turn into the plain assistant-message dict."""

        message: dict[str, Any] = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
                for i, (name, arguments) in enumerate(turn.tool_calls)
            ]
        return message

    # ------------------------------------------------------------- embeddings
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic, order-preserving embeddings for ``texts``."""

        self.embed_calls.append(list(texts))
        return [_deterministic_embedding(text, self._embedding_dim) for text in texts]

    # --------------------------------------------------------------- speech
    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Return a fixed transcript for any audio input (no decoding)."""

        self.transcribe_calls.append(bytes(audio_bytes))
        return "transcribed question about experience"

    async def synthesize(self, text: str) -> bytes:
        """Return deterministic pseudo-audio bytes for ``text``."""

        self.synthesize_calls.append(text)
        if not text or not text.strip():
            return b""
        return b"FAKEMP3" + hashlib.sha256(text.encode("utf-8")).digest()[:8]

    async def aclose(self) -> None:
        """No-op async close, mirroring the real client's shutdown hook."""

        return None


@pytest.fixture()
def fake_llm() -> FakeLLM:
    """Return a default :class:`FakeLLM` (single grounded text answer)."""

    return FakeLLM()


@pytest.fixture()
def make_fake_llm() -> Callable[..., FakeLLM]:
    """Return a factory for building customised :class:`FakeLLM` instances.

    Usage::

        llm = make_fake_llm(scripted_chats=[_ScriptedChat(...)])
    """

    def _factory(**kwargs: Any) -> FakeLLM:
        return FakeLLM(**kwargs)

    return _factory


# Re-export the scripted-turn dataclass so tests can build chat scripts.
ScriptedChat = _ScriptedChat
