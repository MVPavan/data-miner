"""Per-stage-per-image checkpoint system with atomic writes and resume logic.

Layout::

    {base_dir}/{image_id}/detect.json
    {base_dir}/{image_id}/evaluate.json
    {base_dir}/{image_id}/refine.json
    {base_dir}/{image_id}/proposals/{model_name}.json
    {base_dir}/{image_id}/meta.json
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from .contracts import MetaCheckpoint

logger = logging.getLogger("data_miner.auto_annotation_v3.checkpoint")

T = TypeVar("T", bound=BaseModel)

# Canonical pipeline stage order — used for downstream invalidation.
STAGE_ORDER: list[str] = ["detect", "evaluate", "refine"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via a sibling .tmp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages per-image stage checkpoints with atomic writes and resume logic.

    Parameters
    ----------
    base_dir:
        Root directory under which per-image sub-directories are created.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _image_dir(self, image_id: str) -> Path:
        return self.base_dir / image_id

    def _stage_path(self, image_id: str, stage: str) -> Path:
        return self._image_dir(image_id) / f"{stage}.json"

    def _proposals_dir(self, image_id: str) -> Path:
        return self._image_dir(image_id) / "proposals"

    def _meta_path(self, image_id: str) -> Path:
        return self._image_dir(image_id) / "meta.json"

    # ------------------------------------------------------------------
    # Stage checkpoints
    # ------------------------------------------------------------------

    def save(self, image_id: str, stage: str, data: BaseModel) -> Path:
        """Atomically write a stage checkpoint.

        Parameters
        ----------
        image_id:
            Unique identifier for the image (used as the sub-directory name).
        stage:
            Pipeline stage name, e.g. ``"detect"``.
        data:
            Pydantic model instance to serialise.

        Returns
        -------
        Path
            Location of the written file.
        """
        path = self._stage_path(image_id, stage)
        _atomic_write(path, data.model_dump_json(indent=2))
        logger.debug("Saved checkpoint %s/%s → %s", image_id, stage, path)
        return path

    def load(self, image_id: str, stage: str, model_class: Type[T]) -> T | None:
        """Load and validate a stage checkpoint.

        Returns ``None`` if the file does not exist.
        """
        path = self._stage_path(image_id, stage)
        if not path.exists():
            return None
        try:
            return model_class.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load checkpoint %s/%s", image_id, stage)
            return None

    def exists(self, image_id: str, stage: str) -> bool:
        """Return ``True`` if a checkpoint exists for *image_id* / *stage*."""
        return self._stage_path(image_id, stage).exists()

    # ------------------------------------------------------------------
    # Per-model proposal files
    # ------------------------------------------------------------------

    def save_proposal(self, image_id: str, model_name: str, data: BaseModel) -> Path:
        """Save raw per-model proposal results under ``proposals/{model_name}.json``."""
        d = self._proposals_dir(image_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{model_name}.json"
        _atomic_write(path, data.model_dump_json(indent=2))
        logger.debug("Saved proposal %s/%s → %s", image_id, model_name, path)
        return path

    def load_proposal(self, image_id: str, model_name: str, model_class: Type[T]) -> T | None:
        """Load a per-model proposal checkpoint.

        Returns ``None`` if the file does not exist.
        """
        path = self._proposals_dir(image_id) / f"{model_name}.json"
        if not path.exists():
            return None
        try:
            return model_class.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load proposal %s/%s", image_id, model_name)
            return None

    def proposal_exists(self, image_id: str, model_name: str) -> bool:
        """Return ``True`` if a proposal checkpoint exists for *model_name*."""
        return (self._proposals_dir(image_id) / f"{model_name}.json").exists()

    # ------------------------------------------------------------------
    # Meta checkpoint
    # ------------------------------------------------------------------

    def save_meta(self, image_id: str, meta: BaseModel) -> Path:
        """Atomically write the per-image ``meta.json``."""
        path = self._meta_path(image_id)
        _atomic_write(path, meta.model_dump_json(indent=2))
        logger.debug("Saved meta for %s → %s", image_id, path)
        return path

    def load_meta(self, image_id: str) -> MetaCheckpoint | None:
        """Load the per-image ``meta.json``.

        Returns ``None`` if the file does not exist or fails to parse.
        """
        path = self._meta_path(image_id)
        if not path.exists():
            return None
        try:
            return MetaCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load meta for %s", image_id)
            return None

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------

    def should_run_stage(self, image_id: str, stage: str, config_hash: str) -> bool:
        """Decide whether *stage* needs to run.

        A stage must run when:

        * No ``meta.json`` exists yet (first run).
        * The stored ``config_hash`` differs from the current one — in this
          case ``clear_downstream`` is called to invalidate *stage* and every
          later stage.
        * The stage is not listed in ``meta.stages_completed``.
        """
        meta = self.load_meta(image_id)
        if meta is None:
            return True
        if meta.config_hash != config_hash:
            logger.info(
                "Config hash changed for %s (was %s, now %s); invalidating from %s",
                image_id,
                meta.config_hash,
                config_hash,
                stage,
            )
            self.clear_downstream(image_id, stage)
            return True
        return stage not in meta.stages_completed

    def clear_downstream(self, image_id: str, from_stage: str) -> None:
        """Delete *from_stage* and every subsequent stage checkpoint.

        Also removes ``meta.json`` so it is regenerated from scratch.
        """
        try:
            idx = STAGE_ORDER.index(from_stage)
        except ValueError:
            logger.warning("Unknown stage '%s'; skipping clear_downstream.", from_stage)
            return

        for stage in STAGE_ORDER[idx:]:
            path = self._stage_path(image_id, stage)
            if path.exists():
                path.unlink()
                logger.info("Cleared checkpoint %s/%s", image_id, stage)

        meta_path = self._meta_path(image_id)
        if meta_path.exists():
            meta_path.unlink()
            logger.debug("Cleared meta.json for %s", image_id)

    def update_meta(
        self,
        image_id: str,
        stage: str,
        config_hash: str,
        prompt_version: str,
        timing_ms: float,
    ) -> MetaCheckpoint:
        """Record stage completion in ``meta.json``.

        Loads the existing meta (or creates a fresh one), appends *stage* to
        ``stages_completed``, accumulates *timing_ms*, and flips ``status`` to
        ``"complete"`` once all three pipeline stages are done.

        Returns the updated :class:`MetaCheckpoint`.
        """
        meta = self.load_meta(image_id)
        if meta is None:
            meta = MetaCheckpoint(
                image_id=image_id,
                config_hash=config_hash,
                prompt_version=prompt_version,
                status="in_progress",
                stages_completed=[],
                total_timing_ms=0.0,
                final_counts={},
            )

        if stage not in meta.stages_completed:
            meta.stages_completed.append(stage)

        meta.total_timing_ms += timing_ms
        meta.config_hash = config_hash
        meta.prompt_version = prompt_version

        if set(STAGE_ORDER).issubset(set(meta.stages_completed)):
            meta.status = "complete"
        else:
            meta.status = "in_progress"

        self.save_meta(image_id, meta)
        logger.debug(
            "Updated meta for %s: stage=%s completed=%s status=%s timing_ms=%.1f",
            image_id,
            stage,
            meta.stages_completed,
            meta.status,
            meta.total_timing_ms,
        )
        return meta

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def all_stages_complete(self, image_id: str) -> bool:
        """Return ``True`` if all pipeline stages have completed for *image_id*."""
        meta = self.load_meta(image_id)
        return meta is not None and meta.status == "complete"

    def image_ids(self) -> list[str]:
        """Return all image IDs present in the checkpoint directory."""
        return [
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir()
        ]

    def clear_image(self, image_id: str) -> None:
        """Remove all checkpoints for *image_id* (including meta and proposals)."""
        image_dir = self._image_dir(image_id)
        if not image_dir.is_dir():
            return
        # Remove proposal files
        proposals = self._proposals_dir(image_id)
        if proposals.is_dir():
            for f in proposals.iterdir():
                f.unlink(missing_ok=True)
            proposals.rmdir()
        # Remove stage and meta files
        for f in image_dir.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
        try:
            image_dir.rmdir()
        except OSError:
            pass  # non-empty (e.g. nested dirs added externally)
        logger.info("Cleared all checkpoints for %s", image_id)
