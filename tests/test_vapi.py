"""Offline tests for the Vapi telephony bridge (``app/api/routes/vapi.py``).

Fully offline: the brain is faked, nothing touches the network, and no Vapi
account is required. These verify the bridge's OpenAI-compatible contract and its
mapping onto the shared brain:

* streaming (SSE ``chat.completion.chunk`` ... ``[DONE]``) and non-streaming JSON;
* ``messages`` -> ``(query, history)`` mapping, ``channel="voice"``, and the
  best-effort ``conversation_id`` taken from Vapi's ``call.id``;
* the empty-utterance greeting path (brain not invoked);
* shared-secret auth via ``Authorization: Bearer`` and ``x-vapi-secret``.

The brain, retrieval, grounding, and scheduling stack are reused unchanged; only
the thin adapter is exercised here.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_brain, get_settings_dep
from app.brain.persona import BrainResponse
from app.config import Settings
from app.main import create_app

_ANSWER = "I built a hybrid RAG pipeline [1]. It uses BM25 and vector search."


class FakeBrain:
    """Records each ``answer`` call and returns a fixed grounded response."""

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
        self._response.conversation_id = conversation_id or self._response.conversation_id
        return self._response


def _brain_response(answer: str = _ANSWER) -> BrainResponse:
    return BrainResponse(
        answer=answer,
        citations=[],
        tool_calls=[],
        retrieval=None,
        injection_flagged=False,
        grounded=True,
        conversation_id="conv-default",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=12.0,
        latency_breakdown={"total": 12.0},
    )


@pytest.fixture()
def fake_brain() -> FakeBrain:
    return FakeBrain(_brain_response())


def _build_client(fake_brain: FakeBrain, settings: Settings) -> TestClient:
    """Build a TestClient wired to the fake brain and given settings (hermetic).

    Not used as a context manager, so the real lifespan (ChromaDB + Groq client)
    never runs. ``get_settings_dep`` is overridden so the bridge's auth check reads
    the supplied settings rather than the host ``.env``.
    """
    app = create_app()
    app.state.brain = fake_brain
    app.dependency_overrides[get_brain] = lambda: fake_brain
    app.dependency_overrides[get_settings_dep] = lambda: settings
    return TestClient(app)


@pytest.fixture()
def client(fake_brain: FakeBrain, settings: Settings) -> TestClient:
    """Default client with auth disabled (``vapi_secret`` empty in the fixture)."""

    return _build_client(fake_brain, settings)


def _sse_content(text: str) -> str:
    """Concatenate the ``content`` deltas of an OpenAI SSE stream body."""

    pieces: list[str] = []
    saw_done = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            saw_done = True
            continue
        delta = json.loads(data)["choices"][0]["delta"]
        if "content" in delta:
            pieces.append(delta["content"])
    assert saw_done, "stream did not terminate with data: [DONE]"
    return "".join(pieces)


def test_vapi_streaming_returns_brain_answer(client: TestClient, fake_brain: FakeBrain) -> None:
    """Streaming turn returns SSE chunks carrying the brain's answer; brain hit on voice."""

    resp = client.post(
        "/vapi/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "what did you build?"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "hybrid RAG pipeline" in _sse_content(resp.text)

    assert len(fake_brain.calls) == 1
    assert fake_brain.calls[0]["channel"] == "voice"
    assert fake_brain.calls[0]["query"] == "what did you build?"


def test_vapi_non_streaming_returns_chat_completion(client: TestClient) -> None:
    """``stream=false`` returns a single OpenAI ``chat.completion`` JSON object."""

    resp = client.post(
        "/vapi/chat/completions",
        json={"stream": False, "messages": [{"role": "user", "content": "tell me about your experience"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert "hybrid RAG pipeline" in message["content"]


def test_vapi_maps_history_and_conversation_id(client: TestClient, fake_brain: FakeBrain) -> None:
    """Latest user msg becomes the query; prior user/assistant turns become history."""

    resp = client.post(
        "/vapi/chat/completions",
        json={
            "stream": False,
            "call": {"id": "call_abc123"},
            "messages": [
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": "what did you build?"},
                {"role": "assistant", "content": "A RAG pipeline [1]."},
                {"role": "user", "content": "and where did you deploy it?"},
            ],
        },
    )
    assert resp.status_code == 200
    call = fake_brain.calls[0]
    assert call["query"] == "and where did you deploy it?"
    assert call["conversation_id"] == "call_abc123"
    # System + the final user turn are excluded from history.
    assert call["history"] == [
        {"role": "user", "content": "what did you build?"},
        {"role": "assistant", "content": "A RAG pipeline [1]."},
    ]


def test_vapi_empty_utterance_greets_without_calling_brain(
    client: TestClient, fake_brain: FakeBrain
) -> None:
    """No user turn -> a friendly greeting, and the brain is never invoked."""

    resp = client.post(
        "/vapi/chat/completions",
        json={"stream": False, "messages": [{"role": "system", "content": "system only"}]},
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "book a meeting" in content.lower()
    assert fake_brain.calls == []


def test_vapi_auth_enforced_when_secret_set(fake_brain: FakeBrain) -> None:
    """When ``vapi_secret`` is set, the bridge requires the matching secret."""

    secured = Settings(openai_api_key="x", vapi_secret="topsecret")
    c = _build_client(fake_brain, secured)
    payload = {"stream": False, "messages": [{"role": "user", "content": "hi"}]}

    assert c.post("/vapi/chat/completions", json=payload).status_code == 401
    assert (
        c.post(
            "/vapi/chat/completions", json=payload, headers={"Authorization": "Bearer nope"}
        ).status_code
        == 401
    )
    assert (
        c.post(
            "/vapi/chat/completions", json=payload, headers={"Authorization": "Bearer topsecret"}
        ).status_code
        == 200
    )
    # The x-vapi-secret header is also accepted.
    assert (
        c.post(
            "/vapi/chat/completions", json=payload, headers={"x-vapi-secret": "topsecret"}
        ).status_code
        == 200
    )


def test_vapi_events_acks(client: TestClient) -> None:
    """The optional server webhook acks lifecycle messages."""

    resp = client.post("/vapi/events", json={"message": {"type": "end-of-call-report"}})
    assert resp.status_code == 200
    assert resp.json() == {"received": True}
