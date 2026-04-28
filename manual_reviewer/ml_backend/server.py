"""LabelStudioMLBase entry point for the manual_reviewer ML backend.

Thin wrapper over :mod:`manual_reviewer.ml_backend.routes`. The LS ML SDK
is imported lazily so this module remains importable in CI / dev boxes
that don't have ``label_studio_ml`` installed (the tests stub it).

Configuration via env vars:

  AAV4_PIPELINE_DB  required for batch route (cached proposals lookup)
  SAM3_1_URL        SAM 3.1 LitServe endpoint (default: localhost:3014)
  SAM3_1_TIMEOUT    HTTP timeout in seconds (default: 60)
  ML_MODEL_VERSION  passed back to LS as predictions[].model_version
                    (default: ``manual_reviewer_v1``)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from manual_reviewer.ml_backend.aav4_client import build_sam3_client
from manual_reviewer.ml_backend.ls_payload import predictions_envelope
from manual_reviewer.ml_backend.routes import (
    Sam3LikeClient,
    batch_proposals,
    smart_click,
    smart_text,
)

logger = logging.getLogger(__name__)


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
        db_path: Path | str | None = None,
        model_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._sam3_client: Sam3LikeClient | None = sam3_client or _maybe_build_client()
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
                    "score": 1.0 if regions else 0.0,
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
        if isinstance(context, dict) and context.get("result"):
            for region in context["result"]:
                if not isinstance(region, dict):
                    continue
                rtype = (region.get("type") or "").lower()
                if rtype in {"keypointlabels", "keypoint"} and self._sam3_client:
                    return smart_click(task, context, self._sam3_client)
                if rtype == "textarea" and self._sam3_client:
                    return smart_text(task, context, self._sam3_client)
        return batch_proposals(task, self._db_path)

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


def _resolve_db_path(value: Path | str | None) -> Path | None:
    if value is not None:
        p = Path(value)
        return p if p.exists() else None
    env_value = os.environ.get("AAV4_PIPELINE_DB")
    if not env_value:
        return None
    p = Path(env_value)
    return p if p.exists() else None


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
    port = int(os.environ.get("LABEL_STUDIO_ML_PORT", "9090"))
    host = os.environ.get("LABEL_STUDIO_ML_HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    _main()
