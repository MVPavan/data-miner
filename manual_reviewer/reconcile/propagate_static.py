"""Phase C v1: cross-frame static-object propagation.

The reviewer accepts one bbox on a seed image. We crop that bbox, encode
it (DINOv3), and for every scope frame crop at the *same normalized
coords* and compute cosine similarity. Frames whose crop matches the seed
above ``cosine_thresh`` are considered to still hold the same static
instance — we then reconcile against any annotation the reviewer has
already accepted on that frame:

  * existing IoU ≥ ``iou_confirm``       → **confirm**  (silent)
  * ``iou_conflict`` ≤ existing IoU < .. → **weak_iou** (silent, soft flag)
  * existing IoU < ``iou_conflict``      → **conflict** (suggest + flag)
  * no existing rectangle                → **suggest**  (just propose)

Only **reviewer-accepted** annotations count as "existing". Cached
proposals would generate spurious conflicts. Frames where the cosine
falls below threshold are dropped from the pool — they're "no match"
in the doc's table, treated as silent skips.

Static-object only — moving instances are deferred to v2 because sparse
deduped frames break a tracker's motion model and force an appearance-
ReID design with real lookalike risk.

Pure orchestration. No model loading, no HTTP. Inject a
:class:`CropEncoder` for the per-crop embedding (production wires the
DINOv3 wrapper; tests pass a stub) and an existing-rectangles fetcher
(production hits the LS REST API; tests pass a dict).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from manual_reviewer.ml_backend.dedup import iou_xyxy as _shared_iou_xyxy
from manual_reviewer.ml_backend.ls_payload import norm_box_to_ls_region

logger = logging.getLogger(__name__)


__all__ = [
    "CosineGenerator",
    "CropEncoder",
    "Match",
    "PropagateStaticConfig",
    "PropagateStaticSummary",
    "Reconciler",
    "Seed",
    "Verdict",
    "propagate_static",
    "verdict_to_ls_region",
]


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


BboxNorm = tuple[float, float, float, float]   # x1, y1, x2, y2 in [0, 1]


@dataclass(frozen=True)
class Seed:
    """The reviewer's accepted bbox we want to propagate."""

    image_id: str
    image_path: str
    bbox_norm: BboxNorm
    class_name: str


@dataclass(frozen=True)
class Match:
    """A frame whose crop at the seed coords clears the cosine threshold."""

    image_id: str
    image_path: str
    bbox_norm: BboxNorm
    cosine: float


@dataclass(frozen=True)
class Verdict:
    """Reconciler output for one (frame, match) pair.

    ``outcome`` is one of ``confirm`` / ``weak_iou`` / ``conflict`` /
    ``suggest``. Frames that didn't match at all aren't represented here —
    they're aggregated as ``skipped`` on the summary.
    """

    image_id: str
    bbox_norm: BboxNorm
    class_name: str
    cosine: float
    iou_to_existing: float
    outcome: str
    matched_existing_idx: int | None


@dataclass(frozen=True)
class PropagateStaticConfig:
    cosine_thresh: float = 0.85
    iou_confirm: float = 0.7
    iou_conflict: float = 0.4
    max_scope: int = 500
    model_version: str = "propagated_static"


@dataclass
class PropagateStaticSummary:
    confirmed: int = 0
    suggested: int = 0
    weak_iou: int = 0
    conflicts: int = 0
    skipped: int = 0
    out_of_scope: int = 0
    per_image: dict[str, list[Verdict]] = field(default_factory=dict)

    def render_note(self, class_name: str) -> str:
        """Reviewer-facing one-liner appended to the seed task's notes."""
        return (
            f"Propagated {class_name}: "
            f"{self.confirmed} confirmed | "
            f"{self.suggested} suggested | "
            f"{self.conflicts} conflicts | "
            f"{self.weak_iou} weak-iou"
        )


# ---------------------------------------------------------------------------
# Encoder protocol
# ---------------------------------------------------------------------------


class CropEncoder(Protocol):
    """Encode a normalized bbox crop of an image to a vector.

    Implementations are expected to L2-normalize the output (the generator
    re-normalizes defensively in case they don't, so this is best-effort).
    """

    def encode_crop(
        self,
        image_path: str,
        bbox_norm: BboxNorm,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize. Returns the raw vector when norm ≈ 0 (caller's
    responsibility to detect degenerate cases via :func:`_is_degenerate`)."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def _is_degenerate(vec: np.ndarray) -> bool:
    """A near-zero or non-finite embedding — cosine against it is meaningless."""
    if not np.all(np.isfinite(vec)):
        return True
    return float(np.linalg.norm(vec)) <= 1e-12


# Single source of truth for IoU lives in ``ml_backend.dedup``. Re-exported
# here so existing test imports (test_propagate_static imports `_iou_xyxy`
# directly) continue to work and the two helpers can never drift apart.
_iou_xyxy = _shared_iou_xyxy


def _as_bbox_tuple(bbox: Any) -> BboxNorm | None:
    """Coerce a list/tuple of 4 floats into a normalized bbox tuple."""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CosineGenerator:
    """Crop the seed once, then per-frame crop at the same normalized coords.

    Caches the seed embedding so a single :class:`CosineGenerator` instance
    can be reused if the caller batches scope in chunks.
    """

    def __init__(
        self,
        seed: Seed,
        encoder: CropEncoder,
        *,
        cosine_thresh: float = 0.85,
    ) -> None:
        self._seed = seed
        self._encoder = encoder
        self._cosine_thresh = cosine_thresh
        self._seed_emb: np.ndarray | None = None

    def _seed_embedding(self) -> np.ndarray:
        if self._seed_emb is None:
            raw = self._encoder.encode_crop(
                self._seed.image_path, self._seed.bbox_norm
            )
            if _is_degenerate(raw):
                logger.warning(
                    "seed embedding for %s is degenerate "
                    "(non-finite or near-zero norm); every frame will score 0",
                    self._seed.image_id,
                )
            self._seed_emb = _normalize(raw)
        return self._seed_emb

    def generate(
        self,
        scope: list[tuple[str, str]],
    ) -> list[Match]:
        """Return matches for ``scope`` = ``[(image_id, image_path), ...]``.

        Frames whose crop encoding raises are logged and dropped — same
        treatment as a frame whose cosine is below threshold (silent skip
        in the doc's failure-mode table). Non-finite cosines (NaN/Inf
        from a bad embedding) are also dropped: a NaN compare-with-float
        is False either way, so we test ``isfinite`` explicitly to avoid
        a NaN sneaking through as a "match".
        """
        seed_emb = self._seed_embedding()
        out: list[Match] = []
        for image_id, image_path in scope:
            try:
                raw = self._encoder.encode_crop(image_path, self._seed.bbox_norm)
            except Exception as exc:  # noqa: BLE001 — propagation must be best-effort
                logger.warning(
                    "encode_crop failed for image=%s path=%s: %s",
                    image_id,
                    image_path,
                    exc,
                )
                continue
            if _is_degenerate(raw):
                logger.debug(
                    "degenerate embedding for image=%s, dropping", image_id
                )
                continue
            vec = _normalize(raw)
            if seed_emb.shape != vec.shape:
                logger.warning(
                    "embedding dim mismatch for %s (seed=%d, frame=%d), skipping",
                    image_id,
                    seed_emb.shape[0] if seed_emb.ndim == 1 else -1,
                    vec.shape[0] if vec.ndim == 1 else -1,
                )
                continue
            cosine = float(np.dot(seed_emb, vec))
            if not np.isfinite(cosine) or cosine < self._cosine_thresh:
                continue
            out.append(
                Match(
                    image_id=image_id,
                    image_path=image_path,
                    bbox_norm=self._seed.bbox_norm,
                    cosine=cosine,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class Reconciler:
    """Pick an outcome per match by IoU vs. the frame's accepted boxes."""

    def __init__(self, cfg: PropagateStaticConfig) -> None:
        self._cfg = cfg

    def verdict(
        self,
        match: Match,
        existing: list[BboxNorm],
        *,
        class_name: str,
    ) -> Verdict:
        if not existing:
            return Verdict(
                image_id=match.image_id,
                bbox_norm=match.bbox_norm,
                class_name=class_name,
                cosine=match.cosine,
                iou_to_existing=0.0,
                outcome="suggest",
                matched_existing_idx=None,
            )
        ious = [_iou_xyxy(match.bbox_norm, e) for e in existing]
        best_idx = max(range(len(ious)), key=lambda i: ious[i])
        best_iou = ious[best_idx]
        if best_iou >= self._cfg.iou_confirm:
            outcome = "confirm"
        elif best_iou >= self._cfg.iou_conflict:
            outcome = "weak_iou"
        else:
            outcome = "conflict"
        return Verdict(
            image_id=match.image_id,
            bbox_norm=match.bbox_norm,
            class_name=class_name,
            cosine=match.cosine,
            iou_to_existing=best_iou,
            outcome=outcome,
            matched_existing_idx=best_idx,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def propagate_static(
    seed: Seed,
    *,
    scope: list[tuple[str, str]],
    encoder: CropEncoder,
    fetch_existing: Callable[[str], list[BboxNorm]],
    config: PropagateStaticConfig | None = None,
) -> PropagateStaticSummary:
    """Run the full propagate flow; return a :class:`PropagateStaticSummary`.

    **Single-seed by design.** This function takes exactly one ``Seed``
    and produces at most one verdict per scope frame. Callers that need
    multi-seed behaviour (the reviewer accepts two parked bicycles, both
    should propagate) must invoke it once per seed and then NMS the union
    of per-image verdicts before writing back to LS — otherwise the LS
    canvas accumulates duplicate suggestions on overlapping regions and
    the reviewer has to delete them by hand. Built-in multi-seed support
    is parked as a Phase C v2 ticket.

    Args:
      seed: the reviewer's accepted bbox.
      scope: ``[(image_id, image_path), ...]`` — every candidate target frame.
        The seed's own image_id, if present, is filtered out (we don't
        propagate onto the seed frame itself). **Ordering matters when
        ``len(scope) > config.max_scope``**: the cap takes ``scope[:max_scope]``
        with no internal re-sort, so the caller is responsible for putting
        the most-likely candidates first (e.g., DINOv3-nearest survivors
        per plan §C scope discovery). Out-of-cap frames are recorded on
        ``summary.out_of_scope`` rather than skipped.
      encoder: crop-level DINOv3 wrapper.
      fetch_existing: ``image_id -> list[BboxNorm]``. Production wires this
        to the LS REST API (``/api/tasks/{id}/annotations``); tests pass a
        dict-backed callable. Returning ``[]`` is normal — that's the
        "no existing" case and produces a ``suggest`` outcome.
      config: tunables; defaults to :class:`PropagateStaticConfig`.

    Returns a summary with per-image verdicts. The caller is responsible
    for writing predictions back to LS — :func:`verdict_to_ls_region`
    builds the ``RectangleLabels`` payload.
    """
    cfg = config or PropagateStaticConfig()
    summary = PropagateStaticSummary()

    # Drop the seed's own frame from scope — propagating onto itself is a
    # no-op and would generate a confirm-against-itself.
    scope = [(iid, p) for iid, p in scope if iid != seed.image_id]

    if len(scope) > cfg.max_scope:
        summary.out_of_scope = len(scope) - cfg.max_scope
        scope = scope[: cfg.max_scope]
        logger.info(
            "propagate_static: scope capped at %d (dropped %d)",
            cfg.max_scope,
            summary.out_of_scope,
        )

    if not scope:
        return summary

    gen = CosineGenerator(seed, encoder, cosine_thresh=cfg.cosine_thresh)
    matches = gen.generate(scope)

    # Frames in scope but not matched by the generator are "skipped" — they
    # represent the doc's "no match | any → skip" branch.
    matched_ids = {m.image_id for m in matches}
    summary.skipped = sum(1 for iid, _ in scope if iid not in matched_ids)

    rec = Reconciler(cfg)
    for match in matches:
        try:
            existing_raw = fetch_existing(match.image_id) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fetch_existing failed for %s, treating as empty: %s",
                match.image_id,
                exc,
            )
            existing_raw = []
        existing: list[BboxNorm] = []
        for box in existing_raw:
            tup = _as_bbox_tuple(box)
            if tup is not None:
                existing.append(tup)

        verdict = rec.verdict(match, existing, class_name=seed.class_name)
        summary.per_image.setdefault(match.image_id, []).append(verdict)
        if verdict.outcome == "confirm":
            summary.confirmed += 1
        elif verdict.outcome == "weak_iou":
            summary.weak_iou += 1
        elif verdict.outcome == "conflict":
            summary.conflicts += 1
        elif verdict.outcome == "suggest":
            summary.suggested += 1

    logger.info(
        "propagate_static: seed=%s scope=%d matched=%d "
        "confirmed=%d suggested=%d weak_iou=%d conflicts=%d skipped=%d",
        seed.image_id,
        len(scope),
        len(matches),
        summary.confirmed,
        summary.suggested,
        summary.weak_iou,
        summary.conflicts,
        summary.skipped,
    )
    return summary


def verdict_to_ls_region(
    verdict: Verdict,
    *,
    model_version: str = "propagated_static",
) -> dict[str, Any]:
    """Wrap a :class:`Verdict` as an LS ``RectangleLabels`` region.

    ``score`` rides on the LS region as the cosine. ``meta`` carries the
    outcome label so the canvas can tint conflict/weak_iou/confirm
    distinctly if desired.
    """
    return norm_box_to_ls_region(
        list(verdict.bbox_norm),
        verdict.class_name,
        score=verdict.cosine,
        extra_meta={
            "source": "propagate_static",
            "model_version": model_version,
            "cosine": round(verdict.cosine, 4),
            "iou_to_existing": round(verdict.iou_to_existing, 4),
            "outcome": verdict.outcome,
        },
    )
