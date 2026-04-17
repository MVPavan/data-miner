"""Regression tests for the shared HTTP retry helper.

Complements ``tests/test_http_retry.py`` (which verifies policy constants via
an inline replica). These tests verify:

1. The three pipeline stages that do HTTP calls all import the shared helper
   (``workers/http_retry.py``) and use it -- single source of truth.
2. The shared helper's ``http_retry()`` factory actually produces the policy
   verified by the inline replica tests, driven against a real mock server.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from data_miner.auto_annotation_v4.workers.http_retry import (
    _is_transient,
    http_retry,
)


# ---------------------------------------------------------------------------
# 1. Source-level: every stage that does HTTP imports the shared helper.
# ---------------------------------------------------------------------------


STAGE_FILES = [
    Path(__file__).parent.parent / "stages" / "detect_model.py",
    Path(__file__).parent.parent / "stages" / "evaluate.py",
    Path(__file__).parent.parent / "stages" / "refine.py",
]


def test_each_stage_imports_shared_helper():
    for f in STAGE_FILES:
        text = f.read_text()
        assert "from ..workers.http_retry import http_retry" in text, (
            f"{f.name} does not import the shared http_retry helper"
        )
        assert "http_retry()" in text, (
            f"{f.name} imports http_retry but never calls it"
        )


def test_policy_constants_live_in_shared_helper():
    """The tenacity constants should live in exactly one place -- the helper.

    Stages should rely on ``http_retry()`` rather than re-declaring
    ``stop_after_attempt(3)`` / ``wait_exponential(`` inline.
    """
    helper = Path(__file__).parent.parent / "workers" / "http_retry.py"
    helper_text = helper.read_text()
    assert "stop_after_attempt(3)" in helper_text
    assert "wait_exponential(multiplier=1, max=16)" in helper_text
    for f in STAGE_FILES:
        text = f.read_text()
        assert "stop_after_attempt(" not in text, (
            f"{f.name} still has inline stop_after_attempt -- should use http_retry()"
        )
        assert "wait_exponential(" not in text, (
            f"{f.name} still has inline wait_exponential -- should use http_retry()"
        )


# ---------------------------------------------------------------------------
# 2. Functional: the shared helper produces the documented retry behavior.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


class _MockServer:
    def __init__(self, responses):
        self._responses = responses
        self.port = _free_port()
        self.request_count = 0
        self._runner = None

    def _next_status(self) -> int:
        if isinstance(self._responses, int):
            return self._responses
        idx = min(self.request_count - 1, len(self._responses) - 1)
        return self._responses[idx]

    async def _handler(self, _request: web.Request) -> web.Response:
        self.request_count += 1
        status = self._next_status()
        if status == 200:
            return web.json_response({"ok": True})
        return web.Response(status=status, text=f"status {status}")

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/predict", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/predict"


async def _call_with_shared_helper(session: aiohttp.ClientSession, url: str) -> dict:
    retry = http_retry()
    # Short-circuit exponential backoff in unit tests.
    retry.sleep = lambda _s: _no_sleep()
    async for attempt in retry:
        with attempt:
            async with session.post(
                url, json={}, timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
    raise AssertionError("AsyncRetrying exited without attempt")


async def _no_sleep():
    return None


def test_shared_helper_retries_transient_and_recovers():
    async def scenario():
        server = _MockServer([503, 503, 200])
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                data = await _call_with_shared_helper(session, server.url)
            assert data == {"ok": True}
            assert server.request_count == 3
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_shared_helper_no_retry_on_4xx():
    async def scenario():
        server = _MockServer([400])
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                    await _call_with_shared_helper(session, server.url)
            assert exc_info.value.status == 400
            assert server.request_count == 1
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_shared_helper_exhausts_after_three_attempts():
    async def scenario():
        server = _MockServer(503)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                    await _call_with_shared_helper(session, server.url)
            assert exc_info.value.status == 503
            assert server.request_count == 3
        finally:
            await server.stop()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 3. Predicate: _is_transient classifies exceptions correctly.
# ---------------------------------------------------------------------------


def test_is_transient_classification():
    # 4xx -> not transient
    err_400 = aiohttp.ClientResponseError(
        request_info=None, history=(), status=400, message="bad",
    )
    assert _is_transient(err_400) is False

    # 5xx -> transient
    err_503 = aiohttp.ClientResponseError(
        request_info=None, history=(), status=503, message="unavailable",
    )
    assert _is_transient(err_503) is True

    # Connection error -> transient
    assert _is_transient(aiohttp.ClientConnectionError("boom")) is True

    # Timeout -> transient
    assert _is_transient(asyncio.TimeoutError()) is True
    assert _is_transient(TimeoutError()) is True

    # Unrelated exception -> not transient
    assert _is_transient(ValueError("x")) is False
