"""Backend-side adapters: SAM 3.1 HTTP factory + cached-proposals reader.

Two responsibilities:

  1. Construct a :class:`Sam3OneHttpClient` from the ML backend's environment
     (``SAM3_1_URL`` env var or default ``http://localhost:3014/predict``).
  2. Read cached aa_v4 proposals from ``pipeline.db`` for the batch route —
     no inference, just a sqlite SELECT against the ``proposals`` table.

This module does not import label_studio_ml, torch, or sam3. It's safe to
import in tests that don't have the LS ML backend SDK installed.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import requests

from manual_reviewer.reconcile.sam3_client import (
    DEFAULT_SAM3_1_REFINE_URL,
    Sam3OneHttpClient,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_sam3_client",
    "read_cached_proposals",
]


_RETRYABLE_EXCS: tuple[type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
)


class _RetryOnceClient:
    """Wraps :class:`Sam3OneHttpClient` with one retry on transient errors.

    Per-region click latency budget rules out backoff. A single retry on
    ``ConnectionError`` / ``Timeout`` covers the common "server just
    bounced / TCP RST" case without doubling click latency on real
    failures.
    """

    def __init__(self, inner: Sam3OneHttpClient) -> None:
        self._inner = inner

    def _call_with_retry(self, name: str, **kwargs: Any) -> Any:
        method = getattr(self._inner, name)
        try:
            return method(**kwargs)
        except _RETRYABLE_EXCS as exc:
            logger.warning(
                "SAM 3.1 %s: transient error %s; retrying once",
                name,
                exc.__class__.__name__,
            )
            return method(**kwargs)

    def click_mask(self, **kwargs: Any) -> Any:
        return self._call_with_retry("click_mask", **kwargs)

    def text_detect(self, **kwargs: Any) -> Any:
        return self._call_with_retry("text_detect", **kwargs)

    def visual_prompt(self, **kwargs: Any) -> Any:
        return self._call_with_retry("visual_prompt", **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def build_sam3_client(
    *,
    url: str | None = None,
    timeout: float | None = None,
) -> Sam3OneHttpClient:
    """Construct a SAM 3.1 client honoring env-var overrides.

    Env vars:
        SAM3_1_URL: base /predict endpoint (default: localhost:3014).
        SAM3_1_TIMEOUT: per-request timeout in seconds (default: 60).

    Wraps the raw :class:`Sam3OneHttpClient` in a one-retry shim so a
    single transient ConnectionError/Timeout doesn't bubble up to the
    reviewer as a missing region.
    """
    resolved_url = url or os.environ.get("SAM3_1_URL") or DEFAULT_SAM3_1_REFINE_URL
    if timeout is None:
        try:
            timeout = float(os.environ.get("SAM3_1_TIMEOUT", "60"))
        except ValueError:
            timeout = 60.0
    inner = Sam3OneHttpClient(url=resolved_url, timeout=timeout)
    return _RetryOnceClient(inner)  # type: ignore[return-value]


def read_cached_proposals(
    db_path: Path | str,
    image_id: str,
) -> list[dict[str, Any]]:
    """Return every cached proposal row for ``image_id`` as flat candidates.

    Each entry has the union of fields the ML backend needs to seed an LS
    region: ``class_name``, ``confidence``, ``bbox`` (BoundingBox dict or
    None), ``model`` (which detector produced it). Malformed JSON rows are
    skipped silently — the reviewer doesn't need a stack trace on task open.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = TRUE")
        rows = conn.execute(
            "SELECT model, data FROM proposals WHERE image_id = ?",
            (image_id,),
        ).fetchall()
    finally:
        conn.close()

    import json

    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row["data"]
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            out.append(
                {
                    "model": row["model"],
                    "class_name": cand.get("class_name"),
                    "confidence": cand.get("confidence", 0.0),
                    "bbox": cand.get("bbox"),
                    "candidate_id": cand.get("candidate_id"),
                }
            )
    return out
