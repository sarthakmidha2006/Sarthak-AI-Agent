"""API-layer tests using FastAPI's ``TestClient`` (spec §17).

The application's expensive singletons are normally built in the ``lifespan``
context manager (which would construct a ChromaDB-backed vector store and an
OpenAI client). To keep these tests fully offline we deliberately drive the
``TestClient`` **without** entering it as a context manager — so the lifespan
never runs — and instead:

* monkeypatch ``app.state.brain`` to a tiny fake brain whose ``answer`` returns a
  scripted :class:`~app.brain.persona.BrainResponse`;
* set lightweight fakes for ``app.state.vector_store`` / ``app.state.bm25`` so
  ``/health`` can report corpus sizes;
* override the ``get_brain`` and ``get_db`` dependencies so the routes resolve to
  the fake brain and the test database (rebound in :mod:`tests.conftest`).

``/availability`` and ``/book`` exercise the *real* calendar / booking code
against the temp SQLite database; ``/chat`` exercises the route → brain mapping;
``/health`` and the validation-error paths are checked too.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_brain, get_db, get_settings_dep
from app.brain.persona import BrainResponse
from app.config import Settings
from app.db.models import Booking
from app.main import create_app


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeBrain:
    """A stand-in for :class:`~app.brain.persona.PersonaBrain`.

    Records the arguments of each ``answer`` call and returns a pre-built
    :class:`BrainResponse`, so the chat route can be tested without retrieval,
    tools, or the network.
    """

    def __init__(self, response: BrainResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def answer(
        self,
        query: str,
        *,
        channel: str,
        history: list[dict] | None = None,
        conversation_id: str | None = None,
    ) -> BrainResponse:
        self.calls.append(
            {
                "query": query,
                "channel": channel,
                "history": history,
                "conversation_id": conversation_id,
            }
        )
        # Echo the resolved conversation id back so the route surfaces it.
        self._response.conversation_id = conversation_id or self._response.conversation_id
        return self._response


class _FakeVectorStore:
    """Minimal vector-store fake exposing ``count()`` for ``/health``."""

    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _FakeBM25:
    """Minimal BM25 fake exposing a ``size`` property for ``/health``."""

    def __init__(self, size: int) -> None:
        self.size = size


def _brain_response(answer: str = "I have 5 years of experience [1].") -> BrainResponse:
    """Build a representative :class:`BrainResponse`."""

    return BrainResponse(
        answer=answer,
        citations=[
            {
                "n": 1,
                "title": "Resume",
                "source_type": "resume",
                "url": None,
                "snippet": "Five years of backend engineering.",
            }
        ],
        tool_calls=[],
        retrieval=None,
        injection_flagged=False,
        grounded=True,
        conversation_id="conv-123",
        prompt_tokens=42,
        completion_tokens=12,
        latency_ms=15.5,
        latency_breakdown={"total": 15.5},
    )


# --------------------------------------------------------------------------- #
# App / client fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_brain() -> FakeBrain:
    """A fake brain returning a fixed grounded response."""

    return FakeBrain(_brain_response())


@pytest.fixture()
def client(
    fake_brain: FakeBrain,
    settings: Settings,
    session_factory: Callable[[], Any],
    _db_engine,
) -> TestClient:
    """Build a ``TestClient`` wired to fakes and the test database.

    The client is intentionally **not** used as a context manager so the real
    ``lifespan`` (ChromaDB + OpenAI client) never runs. ``app.state`` is populated
    by hand and the relevant dependencies are overridden.
    """

    app = create_app()

    # Populate the state the lifespan would normally set.
    app.state.settings = settings
    app.state.brain = fake_brain
    app.state.vector_store = _FakeVectorStore(count=7)
    app.state.bm25 = _FakeBM25(size=7)

    # Override dependencies: brain → fake, db → test session.
    def _override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_brain] = lambda: fake_brain
    app.dependency_overrides[get_db] = _override_get_db
    # The routes resolve settings via get_settings_dep (the cached host .env),
    # but the date/scheduling math in these tests uses the *pinned* `settings`
    # fixture. Override the dependency so route-time and test-time settings agree
    # (timezone, working hours/days) — otherwise booking a fixture-timezone slot
    # is judged against the host .env timezone and reports "unavailable".
    app.dependency_overrides[get_settings_dep] = lambda: settings

    test_client = TestClient(app)
    return test_client


def _next_working_day(settings: Settings):
    """Return the next strictly-future working day in the business timezone."""

    tz = ZoneInfo(settings.timezone)
    today = datetime.now(tz).date()
    candidate = today + timedelta(days=1)
    working = set(settings.working_days)
    while candidate.weekday() not in working:
        candidate += timedelta(days=1)
    return candidate


def _iso_local(settings: Settings, day, hour: int, minute: int = 0) -> str:
    """ISO-8601 local datetime string on ``day`` at ``hour:minute``."""

    tz = ZoneInfo(settings.timezone)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).isoformat()


# --------------------------------------------------------------------------- #
# /chat
# --------------------------------------------------------------------------- #
def test_chat_happy_path(client: TestClient, fake_brain: FakeBrain) -> None:
    """A valid chat request returns the mapped brain response."""

    resp = client.post("/chat", json={"message": "How many years of experience?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "I have 5 years of experience [1]."
    assert body["session_id"]  # a session id is always returned
    assert body["citations"][0]["title"] == "Resume"
    assert body["tool_calls"] == []
    assert body["injection_flagged"] is False
    assert body["grounded"] is True
    assert body["prompt_tokens"] == 42
    assert body["completion_tokens"] == 12
    # The brain was invoked on the chat channel.
    assert fake_brain.calls[0]["channel"] == "chat"
    assert fake_brain.calls[0]["query"] == "How many years of experience?"


def test_chat_reuses_supplied_session_id(client: TestClient, fake_brain: FakeBrain) -> None:
    """A supplied session_id is forwarded to the brain as the conversation id."""

    resp = client.post(
        "/chat",
        json={"message": "hello", "session_id": "my-session-42"},
    )

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-session-42"
    assert fake_brain.calls[0]["conversation_id"] == "my-session-42"


def test_chat_forwards_history(client: TestClient, fake_brain: FakeBrain) -> None:
    """Conversation history is passed through to the brain as role/content dicts."""

    resp = client.post(
        "/chat",
        json={
            "message": "and after that?",
            "history": [
                {"role": "user", "content": "tell me about your last role"},
                {"role": "assistant", "content": "I was a backend engineer."},
            ],
        },
    )

    assert resp.status_code == 200
    history = fake_brain.calls[0]["history"]
    assert history == [
        {"role": "user", "content": "tell me about your last role"},
        {"role": "assistant", "content": "I was a backend engineer."},
    ]


def test_chat_validation_error_empty_message(client: TestClient) -> None:
    """An empty message violates ``min_length`` and yields HTTP 422."""

    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_validation_error_missing_message(client: TestClient) -> None:
    """A missing ``message`` field yields HTTP 422."""

    resp = client.post("/chat", json={"session_id": "x"})
    assert resp.status_code == 422


def test_chat_validation_error_bad_history_role(client: TestClient) -> None:
    """An invalid history role yields HTTP 422 (Literal validation)."""

    resp = client.post(
        "/chat",
        json={
            "message": "hi",
            "history": [{"role": "system", "content": "not allowed"}],
        },
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /availability
# --------------------------------------------------------------------------- #
def test_availability_happy_path(client: TestClient, settings: Settings) -> None:
    """A valid availability request returns the documented payload shape."""

    day = _next_working_day(settings)
    resp = client.get(
        "/availability",
        params={
            "date_from": day.isoformat(),
            "date_to": day.isoformat(),
            "duration_minutes": 30,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["timezone"] == settings.timezone
    assert body["duration_minutes"] == 30
    assert body["count"] == len(body["slots"])
    # A full empty working day exposes 16 thirty-minute slots.
    assert body["count"] == 16
    assert all({"start", "end"} <= set(slot) for slot in body["slots"])


def test_availability_default_window(client: TestClient, settings: Settings) -> None:
    """Omitting all query params still returns a valid (possibly large) payload."""

    resp = client.get("/availability")
    assert resp.status_code == 200
    body = resp.json()
    assert body["duration_minutes"] == settings.booking_default_duration
    assert isinstance(body["slots"], list)


def test_availability_invalid_date_returns_422(client: TestClient) -> None:
    """A malformed date query param yields HTTP 422."""

    resp = client.get("/availability", params={"date_from": "not-a-date"})
    assert resp.status_code == 422


def test_availability_inverted_range_returns_422(client: TestClient, settings: Settings) -> None:
    """date_to earlier than date_from yields HTTP 422."""

    day = _next_working_day(settings)
    resp = client.get(
        "/availability",
        params={
            "date_from": day.isoformat(),
            "date_to": (day - timedelta(days=2)).isoformat(),
        },
    )
    assert resp.status_code == 422


def test_availability_duration_below_minimum_returns_422(client: TestClient) -> None:
    """A non-positive duration violates the ``ge=1`` constraint (HTTP 422)."""

    resp = client.get("/availability", params={"duration_minutes": 0})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /book
# --------------------------------------------------------------------------- #
def test_book_happy_path(client: TestClient, settings: Settings, session) -> None:
    """A valid booking is confirmed and persisted to the test database."""

    day = _next_working_day(settings)
    resp = client.post(
        "/book",
        json={
            "name": "Carol Example",
            "email": "carol@example.com",
            "start_time": _iso_local(settings, day, 10, 0),
            "duration_minutes": 30,
            "topic": "Coffee chat",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "confirmed"
    assert body["booking_id"]
    assert body["timezone"] == settings.timezone
    assert body["start_time"]
    assert body["alternatives"] is None
    # The booking is in the DB with channel="api".
    booking = session.query(Booking).filter(Booking.id == body["booking_id"]).one()
    assert booking.channel == "api"
    assert booking.email == "carol@example.com"


def test_book_double_book_returns_alternatives(client: TestClient, settings: Settings) -> None:
    """A second booking on the same slot is reported unavailable with alternatives."""

    day = _next_working_day(settings)
    start = _iso_local(settings, day, 11, 0)

    first = client.post(
        "/book",
        json={"name": "First", "email": "first@example.com", "start_time": start},
    )
    assert first.status_code == 200
    assert first.json()["status"] == "confirmed"

    second = client.post(
        "/book",
        json={"name": "Second", "email": "second@example.com", "start_time": start},
    )
    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "unavailable"
    assert body["booking_id"] is None
    assert body["alternatives"]
    assert all({"start", "end"} <= set(alt) for alt in body["alternatives"])


def test_book_invalid_email_returns_422(client: TestClient, settings: Settings) -> None:
    """An invalid email is rejected by ``EmailStr`` at the schema boundary (422)."""

    day = _next_working_day(settings)
    resp = client.post(
        "/book",
        json={
            "name": "Bad Email",
            "email": "not-an-email",
            "start_time": _iso_local(settings, day, 10, 0),
        },
    )
    assert resp.status_code == 422


def test_book_missing_name_returns_422(client: TestClient, settings: Settings) -> None:
    """A missing required ``name`` yields HTTP 422."""

    day = _next_working_day(settings)
    resp = client.post(
        "/book",
        json={
            "email": "ok@example.com",
            "start_time": _iso_local(settings, day, 10, 0),
        },
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /health and /
# --------------------------------------------------------------------------- #
def test_health_reports_corpus_and_models(client: TestClient, settings: Settings) -> None:
    """``/health`` reports liveness, corpus sizes, and configured models."""

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["corpus_chunks"] == 7
    assert body["bm25_size"] == 7
    assert body["models"]["chat"] == settings.openai_chat_model
    assert body["models"]["embedding"] == settings.openai_embedding_model
    assert body["models"]["stt"] == settings.openai_stt_model
    assert body["models"]["tts"] == settings.openai_tts_model


def test_root_info(client: TestClient, settings: Settings) -> None:
    """``GET /`` returns the service-info banner."""

    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "ai-persona"
    assert body["persona"]["name"] == settings.persona_name
    assert "/chat" in body["endpoints"]


def test_process_time_header_present(client: TestClient) -> None:
    """The timing middleware sets the ``X-Process-Time-ms`` response header."""

    resp = client.get("/health")
    assert resp.status_code == 200
    assert "X-Process-Time-ms" in resp.headers
    # Header parses as a float.
    float(resp.headers["X-Process-Time-ms"])
