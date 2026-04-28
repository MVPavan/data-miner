"""Minimal sync HTTP clients for SAM3-family ``/refine`` endpoints.

The reconciler needs one capability today: given an image and a seeded
bbox, return a refined box and confidence. That maps directly to the
existing ``SAM3RefineRequest`` / ``SAM3RefineResponse`` wire contract,
which both ``sam3_dart`` (port 3013) and ``sam3_1`` (port 3014) implement.

Two HTTP client classes ship here:

  * :class:`Sam3HttpClient` — sam3_dart legacy (port 3013).
  * :class:`Sam3OneHttpClient` — SAM 3.1 (port 3014); also exposes
    :meth:`Sam3OneHttpClient.track` for stateless video-tracker calls.

Both implement the :class:`Sam3Client` Protocol used by
``reconcile_group``, so the reconciler is detector-agnostic. Manual review
defaults to SAM 3.1; sam3_dart stays available for cases where SAM 3.1
isn't deployed yet.

We use ``requests`` (sync) rather than aiohttp because ``run_reconcile.py``
is a one-shot CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

from data_miner.auto_annotation_v4.configs.contracts import BoundingBox
from data_miner.auto_annotation_v4.configs.wire import (
    DetectorRequest,
    DetectorResponse,
    SAM3ClickMaskRequest,
    SAM3ClickMaskResponse,
    SAM3RefineRequest,
    SAM3RefineResponse,
    SAM3VideoTrackRequest,
    SAM3VideoTrackResponse,
    SAM3VideoTrackSeed,
)

__all__ = [
    "RefineResponse",
    "Sam3Client",
    "Sam3HttpClient",
    "Sam3OneHttpClient",
    "DEFAULT_SAM3_1_CLICK_URL",
    "DEFAULT_SAM3_1_REFINE_URL",
    "DEFAULT_SAM3_1_TEXT_URL",
    "DEFAULT_SAM3_1_TRACK_URL",
    "DEFAULT_SAM3_DART_REFINE_URL",
]


DEFAULT_SAM3_DART_REFINE_URL = "http://localhost:3013/refine"
DEFAULT_SAM3_1_REFINE_URL = "http://localhost:3014/predict"
DEFAULT_SAM3_1_TRACK_URL = "http://localhost:3014/predict"
DEFAULT_SAM3_1_CLICK_URL = "http://localhost:3014/predict"
DEFAULT_SAM3_1_TEXT_URL = "http://localhost:3014/predict"


@dataclass(frozen=True)
class RefineResponse:
    """Reconciler-facing refine result. ``box`` is None when SAM3-DART had
    nothing confident to return at the seeded location."""

    box: BoundingBox | None
    score: float


class Sam3Client(Protocol):
    """Protocol the reconciler depends on; tests can stub this."""

    def refine(
        self,
        *,
        image_path: str,
        bbox: BoundingBox,
        threshold: float = 0.5,
    ) -> RefineResponse: ...


class _Sam3RefineHttpBase:
    """Shared sync HTTP refine implementation. Subclasses set the default URL."""

    _default_url: str = ""  # override in subclass

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._url = url or self._default_url
        self._timeout = timeout
        self._session = session or requests.Session()

    def refine(
        self,
        *,
        image_path: str,
        bbox: BoundingBox,
        threshold: float = 0.5,
    ) -> RefineResponse:
        req = SAM3RefineRequest(
            image_path=image_path,
            bbox=[bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            threshold=threshold,
        )
        resp = self._session.post(
            self._url,
            json=req.model_dump(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        parsed = SAM3RefineResponse.model_validate(resp.json())
        if not parsed.box or len(parsed.box) != 4:
            return RefineResponse(box=None, score=parsed.score)
        x1, y1, x2, y2 = parsed.box
        return RefineResponse(
            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            score=parsed.score,
        )


class Sam3HttpClient(_Sam3RefineHttpBase):
    """SAM3-DART HTTP client (legacy default, port 3013).

    Kept for back-compat / fallback when SAM 3.1 isn't deployed.
    """

    _default_url = DEFAULT_SAM3_DART_REFINE_URL


class Sam3OneHttpClient(_Sam3RefineHttpBase):
    """SAM 3.1 HTTP client (manual_reviewer default, port 3014).

    Wraps the four LitAPI dispatch shapes:

      * :meth:`refine` — bbox → refined bbox + score (inherited)
      * :meth:`track` — stateless video-tracker RPC
      * :meth:`click_mask` — point → mask + tight bbox (LS smart_click)
      * :meth:`text_detect` — text → boxes + scores (LS smart_text)

    Same ``Sam3Client`` Protocol surface for ``refine``, so the reconciler
    treats this as a drop-in replacement for ``Sam3HttpClient``.
    """

    _default_url = DEFAULT_SAM3_1_REFINE_URL

    def __init__(
        self,
        url: str | None = None,
        *,
        track_url: str | None = None,
        click_url: str | None = None,
        text_url: str | None = None,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(url=url, timeout=timeout, session=session)
        # SAM 3.1 LitServe uses one endpoint for all modes (mode-dispatch
        # by request shape); per-mode URLs default to the same endpoint.
        self._track_url = track_url or url or DEFAULT_SAM3_1_TRACK_URL
        self._click_url = click_url or url or DEFAULT_SAM3_1_CLICK_URL
        self._text_url = text_url or url or DEFAULT_SAM3_1_TEXT_URL

    def track(
        self,
        *,
        resource_path: str,
        seeds: list[SAM3VideoTrackSeed],
        propagation_direction: str = "both",
        max_frames: int | None = None,
        return_masks: bool = False,
    ) -> SAM3VideoTrackResponse:
        """Stateless video-tracker RPC — server manages SAM 3.1 sessions."""
        req = SAM3VideoTrackRequest(
            resource_path=resource_path,
            seeds=seeds,
            propagation_direction=propagation_direction,
            max_frames=max_frames,
            return_masks=return_masks,
        )
        resp = self._session.post(
            self._track_url,
            json=req.model_dump(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return SAM3VideoTrackResponse.model_validate(resp.json())

    def click_mask(
        self,
        *,
        image_path: str,
        point: list[float],
        point_label: int = 1,
        threshold: float = 0.5,
    ) -> SAM3ClickMaskResponse:
        """Point → mask inference for the LS ML backend smart_click route."""
        req = SAM3ClickMaskRequest(
            image_path=image_path,
            point=point,
            point_label=point_label,
            threshold=threshold,
        )
        resp = self._session.post(
            self._click_url,
            json=req.model_dump(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return SAM3ClickMaskResponse.model_validate(resp.json())

    def text_detect(
        self,
        *,
        image_path: str,
        prompts: list[str],
        threshold: float | None = None,
    ) -> DetectorResponse:
        """Text → detect inference for the LS ML backend smart_text route."""
        req = DetectorRequest(
            image_path=image_path,
            prompts=prompts,
            threshold=threshold,
        )
        resp = self._session.post(
            self._text_url,
            json=req.model_dump(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return DetectorResponse.model_validate(resp.json())
