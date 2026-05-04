"""LS REST helpers for cross-task writes from the ML backend.

The smart-tool routes that propagate predictions across sibling tasks
(currently :func:`smart_track`) need to:

  1. Find sibling tasks in the same project given a clip-prefix filter.
  2. POST predictions to those tasks via ``/api/predictions/`` so the
     reviewer sees them on task open.

The sync side (``scripts/sync_ls_to_disk.py``) uses a one-shot httpx.Client
inside a function. Here we keep a long-lived client on the backend
process — predict() may be called many times per second, and re-creating
the connection pool each call is wasteful.

Configure via env vars (set by ``manage_stack.sh``):

  LS_URL    base URL, e.g. http://localhost:8080
  LS_TOKEN  user API token

If either is missing, :class:`LSRestClient` raises on construction so
callers can fall back gracefully.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "LSRestClient",
    "build_ls_rest_client",
]


class LSRestClient:
    """Thin wrapper around httpx.Client targeting the LS REST API.

    Methods are scoped to what the ML backend needs — task lookup and
    prediction writes. Not a general-purpose LS SDK.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Token {token}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LSRestClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def iter_project_tasks(
        self,
        project_id: int,
        *,
        page_size: int = 200,
    ) -> Iterator[dict[str, Any]]:
        """Yield every task in a project (paginated, ``fields=all``).

        ``fields=all`` is required so each task carries ``data`` (where
        ``image_id`` and ``image_path`` live) — without it, LS returns a
        terse summary and we can't filter by clip prefix.
        """
        page = 1
        while True:
            resp = self._client.get(
                f"/api/projects/{project_id}/tasks",
                params={"page": page, "page_size": page_size, "fields": "all"},
            )
            # LS returns 404 (not an empty list) when paginating past the
            # last page on totals divisible by page_size — treat as EOF.
            if resp.status_code == 404:
                return
            resp.raise_for_status()
            data = resp.json()
            tasks = data if isinstance(data, list) else (
                data.get("tasks") or data.get("results") or []
            )
            if not tasks:
                return
            for task in tasks:
                if isinstance(task, dict):
                    yield task
            if len(tasks) < page_size:
                return
            page += 1

    def post_prediction(
        self,
        *,
        task_id: int,
        result: list[dict[str, Any]],
        score: float,
        model_version: str,
    ) -> int | None:
        """POST a prediction; return its id, or None on failure.

        We never raise on this path — propagation that fails for one
        sibling shouldn't kill the whole call. Caller logs the count and
        moves on.
        """
        try:
            resp = self._client.post(
                "/api/predictions/",
                json={
                    "task": task_id,
                    "result": result,
                    "score": float(score),
                    "model_version": model_version,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "post_prediction failed for task %s: %s", task_id, exc
            )
            return None
        body = resp.json() if resp.content else {}
        if isinstance(body, dict):
            pid = body.get("id")
            if isinstance(pid, int):
                return pid
        return None

    def list_predictions(self, task_id: int) -> list[dict[str, Any]]:
        """Return every prediction attached to ``task_id``.

        LS exposes predictions inline on /api/tasks/<id>/, which is the
        single-call way to fetch them. Returns ``[]`` on any HTTP /
        parse error so callers can default to "no existing predictions"
        without crashing.
        """
        try:
            resp = self._client.get(f"/api/tasks/{task_id}/")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "list_predictions failed for task %s: %s", task_id, exc
            )
            return []
        body = resp.json() if resp.content else {}
        preds = body.get("predictions") if isinstance(body, dict) else None
        return [p for p in (preds or []) if isinstance(p, dict)]

    def patch_prediction(
        self,
        prediction_id: int,
        *,
        result: list[dict[str, Any]],
        score: float | None = None,
        model_version: str | None = None,
    ) -> bool:
        """PATCH ``/api/predictions/<id>/`` with a new result list.

        Returns True on 2xx, False otherwise. Like ``post_prediction``,
        never raises — partial-failure tolerance for batch propagation.
        """
        payload: dict[str, Any] = {"result": result}
        if score is not None:
            payload["score"] = float(score)
        if model_version is not None:
            payload["model_version"] = model_version
        try:
            resp = self._client.patch(
                f"/api/predictions/{prediction_id}/", json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "patch_prediction failed for prediction %s: %s",
                prediction_id, exc,
            )
            return False
        return True

    def delete_prediction(self, prediction_id: int) -> bool:
        """DELETE ``/api/predictions/<id>/``. Returns True on success."""
        try:
            resp = self._client.delete(f"/api/predictions/{prediction_id}/")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "delete_prediction failed for prediction %s: %s",
                prediction_id, exc,
            )
            return False
        return True

    def delete_annotation(self, annotation_id: int) -> bool:
        """DELETE ``/api/annotations/<id>/``. Returns True on success."""
        try:
            resp = self._client.delete(f"/api/annotations/{annotation_id}/")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "delete_annotation failed for annotation %s: %s",
                annotation_id, exc,
            )
            return False
        return True


def build_ls_rest_client(
    *,
    base_url: str | None = None,
    token: str | None = None,
    timeout: float | None = None,
) -> LSRestClient | None:
    """Construct an LSRestClient from env, or return None if unconfigured.

    The ML backend must keep working when LS_URL/LS_TOKEN aren't set
    (e.g. during unit tests or a degraded deploy). Routes that need
    cross-task writes check the return for None and skip gracefully.
    """
    url = base_url or os.environ.get("LS_URL")
    tok = token or os.environ.get("LS_TOKEN")
    if not url or not tok:
        return None
    if timeout is None:
        try:
            timeout = float(os.environ.get("LS_TIMEOUT", "30"))
        except ValueError:
            timeout = 30.0
    return LSRestClient(base_url=url, token=tok, timeout=timeout)
