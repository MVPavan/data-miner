"""Tier 1 unit tests for DetectModelWorker HTTP retry policy.

The policy under test is defined inline in
``auto_annotation_v4/stages/detect_model.py::DetectModelWorker._call_model``:

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=16),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    ):
        with attempt:
            async with self._session.post(url, json=req.model_dump(), ...) as resp:
                resp.raise_for_status()
                data = await resp.json()

    where _is_transient() returns False for aiohttp.ClientResponseError with
    400 <= status < 500, and True for other aiohttp.ClientError / TimeoutError.

We verify the policy directly (approach "c" from the test brief): a thin
aiohttp POST wrapped in an AsyncRetrying with identical constants and the
same _is_transient predicate, pointed at a real aiohttp.web mock server.
This proves the policy itself behaves correctly without invoking full worker
lifecycle (CheckpointDB, semaphores, class loading). ``sleep=lambda _: None``
on AsyncRetrying keeps total runtime <1s.
"""

from __future__ import annotations

import asyncio
import socket

import aiohttp
from aiohttp import web
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def _run(coro):
    return asyncio.run(coro)


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _is_transient(exc: BaseException) -> bool:
    """Mirror of DetectModelWorker._call_model._is_transient."""
    if isinstance(exc, aiohttp.ClientResponseError) and 400 <= exc.status < 500:
        return False
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError))


async def _no_sleep(_seconds: float) -> None:
    """AsyncRetrying awaits the sleep callable; return an awaited no-op."""
    return None


async def _post_with_retry(session: aiohttp.ClientSession, url: str) -> dict:
    """Replica of the retry-wrapped POST in DetectModelWorker._call_model.

    Identical stop/wait/retry constants. ``sleep=_no_sleep`` short-circuits
    the exponential backoff for unit-test speed (the real worker does not do
    this; here it's a test-only override).
    """
    request_timeout = aiohttp.ClientTimeout(total=5)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=16),
        retry=retry_if_exception(_is_transient),
        reraise=True,
        sleep=_no_sleep,
    ):
        with attempt:
            async with session.post(
                url, json={"hello": "world"}, timeout=request_timeout,
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
    raise AssertionError("AsyncRetrying exited without attempt")


class _MockServer:
    """Tiny aiohttp mock server that sequences canned responses per-request."""

    def __init__(self, responses: list[int] | int) -> None:
        self._responses = responses
        self.port = _free_port()
        self.request_count = 0
        self._runner: web.AppRunner | None = None

    def _next_status(self) -> int:
        if isinstance(self._responses, int):
            return self._responses
        idx = min(self.request_count - 1, len(self._responses) - 1)
        return self._responses[idx]

    async def _handler(self, _request: web.Request) -> web.Response:
        self.request_count += 1
        status = self._next_status()
        if status == 200:
            return web.json_response({"boxes": [], "scores": [], "labels": []})
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


# ---------------------------------------------------------------------------
# 1. Transient 503 -> 503 -> 200 recovers on the 3rd attempt
# ---------------------------------------------------------------------------


def test_transient_failure_recovers_on_third_attempt():
    async def scenario():
        server = _MockServer([503, 503, 200])
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                data = await _post_with_retry(session, server.url)
            assert data == {"boxes": [], "scores": [], "labels": []}
            assert server.request_count == 3, (
                f"Expected 3 requests, got {server.request_count}"
            )
        finally:
            await server.stop()

    _run(scenario())


# ---------------------------------------------------------------------------
# 2. 4xx status -> no retry, fail immediately
# ---------------------------------------------------------------------------


def test_no_retry_on_4xx():
    async def scenario():
        server = _MockServer([400])
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                raised: aiohttp.ClientResponseError | None = None
                try:
                    await _post_with_retry(session, server.url)
                except aiohttp.ClientResponseError as e:
                    raised = e
            assert raised is not None, "Expected ClientResponseError on 4xx"
            assert raised.status == 400
            assert server.request_count == 1, (
                f"Expected exactly 1 request on 4xx, got {server.request_count}"
            )
        finally:
            await server.stop()

    _run(scenario())


# ---------------------------------------------------------------------------
# 3. Persistent 503 -> retry exhausts after 3 attempts, raises
# ---------------------------------------------------------------------------


def test_max_retries_exhausted():
    async def scenario():
        server = _MockServer(503)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                raised: aiohttp.ClientResponseError | None = None
                try:
                    await _post_with_retry(session, server.url)
                except aiohttp.ClientResponseError as e:
                    raised = e
            assert raised is not None, "Expected ClientResponseError after exhausted retries"
            assert raised.status == 503
            assert server.request_count == 3, (
                f"Expected 3 attempts before giving up, got {server.request_count}"
            )
        finally:
            await server.stop()

    _run(scenario())
