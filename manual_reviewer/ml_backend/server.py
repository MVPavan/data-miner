"""LabelStudioMLBase entry point for the manual_reviewer ML backend.

Thin wrapper over :mod:`manual_reviewer.ml_backend.routes`. The LS ML SDK
is imported lazily so this module remains importable in CI / dev boxes
that don't have ``label_studio_ml`` installed (the tests stub it).

Configuration via env vars:

  AAV4_PIPELINE_DB         required for batch route (cached proposals lookup)
  SAM3_1_URL               SAM 3.1 LitServe endpoint (default: localhost:3014)
  SAM3_1_TIMEOUT           HTTP timeout in seconds (default: 60)
  ML_MODEL_VERSION         passed back to LS as predictions[].model_version
                           (default: ``manual_reviewer_v1``)

Per-route gates (Phase D-toggles). Default all enabled. Restart-to-toggle
is fine for v1; we don't need a runtime endpoint with one reviewer.

  ENABLE_BATCH_PROPOSALS   task-open cached-proposals seed
  ENABLE_SMART_CLICK       KeyPoint draft → SAM 3.1 click_mask
  ENABLE_SMART_SEARCH      TextArea submit → SAM 3.1 text_detect
  ENABLE_SMART_VISUAL      smart-Rectangle draft → SAM 3.1 visual_prompt
  ENABLE_SMART_TRACK       smart-Rectangle draft → SAM 3.1 video tracker
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from manual_reviewer.ml_backend.aav4_client import build_sam3_client
from manual_reviewer.ml_backend.ls_payload import predictions_envelope
from manual_reviewer.ml_backend.ls_rest import LSRestClient, build_ls_rest_client
from manual_reviewer.ml_backend.routes import (
    Sam3LikeClient,
    batch_proposals,
    smart_click,
    smart_search,
    smart_track,
    smart_visual,
)

logger = logging.getLogger(__name__)


_ROUTE_ENV_VARS = {
    "batch_proposals": "ENABLE_BATCH_PROPOSALS",
    "smart_click": "ENABLE_SMART_CLICK",
    "smart_search": "ENABLE_SMART_SEARCH",
    "smart_track": "ENABLE_SMART_TRACK",
    "smart_visual": "ENABLE_SMART_VISUAL",
}


def _route_enabled(name: str) -> bool:
    """True unless the route's gate env var is set to a falsy value.

    Falsy spellings (case-insensitive): ``0``, ``false``, ``no``, ``off``,
    ``disabled``, ``""``. Anything else (including unset) keeps the route
    enabled — defaults are open so the reviewer doesn't have to set five
    env vars to get the working set back.
    """
    env_name = _ROUTE_ENV_VARS.get(name)
    if env_name is None:
        return True
    raw = os.environ.get(env_name)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled", ""}


def _load_ls_base() -> type:
    """Lazy-import LabelStudioMLBase; return ``object`` if unavailable.

    Letting the class fall back to ``object`` keeps ``import server`` working
    in test environments without the LS SDK. The CLI entry point at the
    bottom requires the real base class.
    """
    try:
        from label_studio_ml.model import LabelStudioMLBase  # type: ignore[import-not-found]

        return LabelStudioMLBase
    except Exception as exc:  # noqa: BLE001
        logger.debug("label_studio_ml not available, using stub base: %s", exc)
        return object


_LSBase = _load_ls_base()


class ManualReviewerMLBackend(_LSBase):  # type: ignore[misc, valid-type]
    """ML backend that adapts SAM 3.1 + cached aa_v4 proposals to LS predict().

    Three modes are picked by request shape; see :mod:`routes` for the
    decision logic.
    """

    def __init__(
        self,
        *,
        sam3_client: Sam3LikeClient | None = None,
        ls_rest: LSRestClient | None = None,
        db_path: Path | str | None = None,
        model_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._sam3_client: Sam3LikeClient | None = sam3_client or _maybe_build_client()
        self._ls_rest: LSRestClient | None = ls_rest or build_ls_rest_client()
        self._db_path: Path | None = _resolve_db_path(db_path)
        self._model_version = model_version or os.environ.get(
            "ML_MODEL_VERSION", "manual_reviewer_v1"
        )

    # ------------------------------------------------------------------
    # LS hook
    # ------------------------------------------------------------------

    def predict(  # type: ignore[override]
        self,
        tasks: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return one predictions envelope per task.

        LS supports either smart-tool drafts (one task at a time, a non-empty
        ``context``) or task-open seeding (many tasks, no ``context``). We
        handle both by routing each task individually.
        """
        out: list[dict[str, Any]] = []
        for task in tasks:
            try:
                regions = self._predict_one(task, context)
            except Exception as exc:  # noqa: BLE001 — never crash LS
                logger.exception("predict failed for task %s: %s", task.get("id"), exc)
                regions = []
            out.append(
                {
                    "result": regions,
                    "model_version": self._model_version,
                    "score": max(
                        (float(r.get("score", 0.0) or 0.0) for r in regions),
                        default=0.0,
                    ),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_one(
        self,
        task: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        ctx_summary = self._summarize_context(context)
        logger.info(
            "predict task=%s context=%s sam3=%s db=%s",
            task.get("id"),
            ctx_summary,
            "yes" if self._sam3_client else "no",
            "yes" if self._db_path else "no",
        )
        has_draft = isinstance(context, dict) and bool(context.get("result"))
        if has_draft:
            # smart_track is discriminated by ``from_name="smart_track"`` —
            # check it first so a smart_track draft never falls into the
            # generic rectangle handlers below.
            has_track = self._sam3_client and any(
                isinstance(r, dict) and r.get("from_name") == "smart_track"
                for r in context["result"]
            )
            if has_track and _route_enabled("smart_track"):
                _, propagate_result = smart_track(
                    task, context, self._sam3_client, self._ls_rest,
                )
                if propagate_result is None:
                    logger.info("→ smart_track did not run (preconditions unmet)")
                else:
                    logger.info(
                        "→ smart_track propagated to %d/%d siblings (rejected motion=%d, score=%d, missing=%d)",
                        propagate_result.propagated,
                        propagate_result.siblings_total,
                        propagate_result.rejected_motion,
                        propagate_result.rejected_score,
                        propagate_result.rejected_missing,
                    )
                return []
            if has_track:
                logger.info("→ smart_track disabled by env, falling through")
            # smart_visual is discriminated by the smart-Rectangle's
            # from_name, not by region type — a regular bbox draw also
            # produces type "rectanglelabels" and we don't want THAT to
            # fire ML.
            has_visual = self._sam3_client and any(
                isinstance(r, dict) and r.get("from_name") == "smart_visual"
                for r in context["result"]
            )
            if has_visual and _route_enabled("smart_visual"):
                out = smart_visual(task, context, self._sam3_client)
                logger.info("→ smart_visual returned %d region(s)", len(out))
                return out
            if has_visual:
                logger.info("→ smart_visual disabled by env, falling through")
            # `continue` (not `return []`) on a gated-off branch so a mixed
            # context (e.g. textarea + keypoint) with one disabled route +
            # one enabled route still reaches the enabled route below.
            # Same applies when smart_visual is gated off but a keypoint
            # or textarea also rides on the same draft.
            for region in context["result"]:
                if not isinstance(region, dict):
                    continue
                if region.get("from_name") in {"smart_visual", "smart_track"}:
                    # Smart tools with their own dispatch branches above —
                    # never let them fall through into the generic
                    # smart_click / smart_search handlers. Both produce
                    # type=rectanglelabels which would otherwise match the
                    # generic paths.
                    continue
                rtype = (region.get("type") or "").lower()
                if rtype in {"keypointlabels", "keypoint"} and self._sam3_client:
                    if not _route_enabled("smart_click"):
                        logger.info("→ smart_click disabled by env, skipping draft")
                        continue
                    out = smart_click(task, context, self._sam3_client)
                    logger.info("→ smart_click returned %d region(s)", len(out))
                    return out
                if rtype == "textarea" and self._sam3_client:
                    if not _route_enabled("smart_search"):
                        logger.info("→ smart_search disabled by env, skipping draft")
                        continue
                    out = smart_search(task, context, self._sam3_client)
                    logger.info("→ smart_search returned %d region(s)", len(out))
                    return out
            # We had a draft but no enabled smart route handled it. Don't
            # fall through to batch_proposals — that would seed the canvas
            # with cached proposals after a click/textarea, which the
            # reviewer didn't ask for.
            logger.info("→ no enabled smart route matched draft, returning empty")
            return []
        if not _route_enabled("batch_proposals"):
            logger.info("→ batch_proposals disabled by env, skipping")
            return []
        out = batch_proposals(task, self._db_path)
        logger.info("→ batch_proposals returned %d region(s)", len(out))
        return out

    @staticmethod
    def _summarize_context(context: Any) -> str:
        if not isinstance(context, dict):
            return f"<{type(context).__name__}>"
        result = context.get("result")
        if not isinstance(result, list):
            return f"keys={sorted(context.keys())}"
        # Include from_name so we can tell smart_visual draft from regular
        # bbox draws when both produce type=rectanglelabels.
        items = [
            f"{(r or {}).get('from_name')}:{(r or {}).get('type')}"
            for r in result if isinstance(r, dict)
        ]
        return f"results=[{', '.join(items)}]"

    # Public envelope helper for callers that bypass LS but want the same
    # ``predictions`` shape (used by the test suite).

    def envelope(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return predictions_envelope(regions, model_version=self._model_version)


def _maybe_build_client() -> Sam3LikeClient | None:
    """Build a SAM 3.1 client only if the env var is set or default is OK.

    In bare test environments the network call won't succeed; we still let
    the client object exist (its methods are only called when smart routes
    fire) and rely on the routes' try/except to swallow connection errors.
    """
    try:
        return build_sam3_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not build SAM 3.1 client: %s", exc)
        return None


_DB_PATH_WARNED: set[str] = set()


def _resolve_db_path(value: Path | str | None) -> Path | None:
    if value is not None:
        p = Path(value)
        return p if p.exists() else None
    env_value = os.environ.get("AAV4_PIPELINE_DB")
    if not env_value:
        return None
    p = Path(env_value)
    if p.exists():
        return p
    if env_value not in _DB_PATH_WARNED:
        logger.warning(
            "AAV4_PIPELINE_DB=%s does not exist; batch_proposals will return empty",
            env_value,
        )
        _DB_PATH_WARNED.add(env_value)
    return None


# ---------------------------------------------------------------------------
# CLI entry point — mirrors LS ML backend convention
# ---------------------------------------------------------------------------


def _main() -> None:
    """Run the LS ML backend HTTP server.

    Requires ``label_studio_ml`` installed. The server picks a port from the
    ``LABEL_STUDIO_ML_PORT`` env var (default 9090).
    """
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        from label_studio_ml.api import init_app  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — runtime-only
        raise SystemExit(
            "label_studio_ml is not installed; "
            "pip install label-studio-ml to run the backend"
        ) from exc

    app = init_app(model_class=ManualReviewerMLBackend)

    # Register the LS annotation backup endpoint (push-based on-disk
    # capture; lossless audit + per-annotation snapshots). Independent
    # of pipeline.db; configure a webhook in LS pointing at this URL.
    from manual_reviewer.ml_backend.lswebhook import register_lswebhook_routes
    register_lswebhook_routes(app)

    port = int(os.environ.get("LABEL_STUDIO_ML_PORT", "9090"))
    host = os.environ.get("LABEL_STUDIO_ML_HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    _main()
