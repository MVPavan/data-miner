"""Shared HTTP retry policy for pipeline stages that call model servers.

All stages that POST to a model server (detector, VLM, SAM3) share the same
transient-failure retry contract:

    stop_after_attempt(3), wait_exponential(multiplier=1, max=16)

``_is_transient`` returns False for aiohttp.ClientResponseError with
400 <= status < 500 (client/code bug -- retrying wastes time and floods the
server) and True for aiohttp.ClientError / asyncio.TimeoutError / TimeoutError
(transient network / server 5xx / timeout).

Usage pattern mirrors the one verified by tests/test_http_retry.py::

    async for attempt in http_retry():
        with attempt:
            async with session.post(url, json=payload, timeout=...) as resp:
                resp.raise_for_status()
                data = await resp.json()
"""

from __future__ import annotations

import asyncio

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def _is_transient(exc: BaseException) -> bool:
    """True iff the exception should trigger a retry."""
    if isinstance(exc, aiohttp.ClientResponseError) and 400 <= exc.status < 500:
        return False
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError))


def http_retry() -> AsyncRetrying:
    """Return a fresh ``AsyncRetrying`` with the shared policy constants."""
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=16),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )
