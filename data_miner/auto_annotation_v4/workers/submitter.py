"""JobSubmitter — smart work queue seeding for the v4 pipeline.

Phase 2 enhancements:
- Per-model detect queuing (``"detect:{model}"`` stage keys)
- Stage-aware submission (skip stages not in ``runtime.stages``)
- Force controls: ``force_rerun``, ``force_stages``, ``force_detect_models``
- Proposal cache awareness (skip models that already have proposals)
- Barrier-aware: queues ``"detect:merge"`` directly when all proposals cached

Usage::

    async with CheckpointDB(db_path) as db:
        submitter = JobSubmitter(config, db)
        submitted, total = await submitter.submit_directory(
            "/data/images", job_id="run_001"
        )
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..checkpoint import CheckpointDB
from ..configs.enums import DetectorName, Stage
from ..configs.settings import AutoAnnotationV4Config

logger = logging.getLogger("data_miner.auto_annotation_v4.submitter")


class JobSubmitter:
    """Submits images to the pipeline work queue with Phase 2 smart routing.

    Handles force controls, per-model detect queueing, proposal caching,
    and stage-aware submission.  Skips already-complete images for resume.

    Args:
        config: Pipeline configuration.
        db: CheckpointDB instance (must be connected).
    """

    _IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    )

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        db: CheckpointDB,
    ) -> None:
        self.config = config
        self.db = db

    # ------------------------------------------------------------------
    # Internal: resolve which models to target
    # ------------------------------------------------------------------

    def _target_models(self) -> list[DetectorName]:
        """Determine which detectors to queue work for.

        If ``runtime.detect_models`` is non-empty, use that subset.
        Otherwise, use all enabled detectors from ``servers.detectors``.
        """
        enabled = self.config.servers.enabled_detectors()
        if self.config.runtime.detect_models:
            return [m for m in self.config.runtime.detect_models if m in enabled]
        return list(enabled.keys())

    # ------------------------------------------------------------------
    # Internal: per-image force handling
    # ------------------------------------------------------------------

    async def _apply_force_controls(self, image_id: str) -> None:
        """Apply force_rerun, force_stages, force_detect_models for one image.

        Called before queueing work.  Mutates the DB state to clear
        stale data so workers will re-process the affected stages.
        """
        rt = self.config.runtime

        # Nuclear: clear everything for this image
        if rt.force_rerun:
            await self.db.clear_image(image_id)
            return  # nothing else to do after clearing everything

        # Force specific stages: clear from that stage onward
        for stage in rt.force_stages:
            await self.db.clear_downstream(image_id, stage)

        # Force specific detect models: delete their proposals + detect stage + downstream
        if rt.force_detect_models:
            for model in rt.force_detect_models:
                await self.db.delete_proposal(image_id, model.value)
            # Detect stage result is stale if any model's proposal was deleted
            await self.db.delete_stage(image_id, Stage.DETECT.value)
            await self.db.clear_downstream(image_id, Stage.DETECT)

    # ------------------------------------------------------------------
    # Internal: queue detect work for one image
    # ------------------------------------------------------------------

    async def _queue_detect_work(self, image_id: str) -> None:
        """Queue per-model detect work items for one image.

        For each target model:
        - Skip if proposal already cached (unless forced)
        - Queue ``"detect:{model.value}"`` work item

        After queueing, check if the barrier is already met (all proposals
        cached from a previous run) and queue ``"detect:merge"`` directly.
        """
        target_models = self._target_models()
        models_queued = 0

        for model in target_models:
            # Skip if proposal already cached
            if await self.db.proposal_exists(image_id, model):
                logger.debug(
                    "Proposal %s/%s already cached — skipping",
                    image_id, model.value,
                )
                continue
            stage_key = f"detect:{model.value}"
            await self.db.add_work(stage_key, image_id)
            models_queued += 1

        # If all proposals already cached, queue merge directly
        if models_queued == 0:
            model_values = [m.value for m in target_models]
            if await self.db.barrier_ready(image_id, model_values):
                # Check if detect stage result exists — if it does, no need to merge
                if not await self.db.stage_exists(image_id, Stage.DETECT):
                    await self.db.add_work("detect:merge", image_id)
                    logger.debug(
                        "All proposals cached for %s — queued detect:merge",
                        image_id,
                    )

    # ------------------------------------------------------------------
    # Internal: queue non-detect first stage
    # ------------------------------------------------------------------

    async def _queue_first_stage(
        self, image_id: str, first_stage: Stage
    ) -> bool:
        """Queue work for a non-detect first stage (e.g. evaluate, refine).

        Checks that the prerequisite stage exists before queueing.

        Returns:
            True if work was queued, False if prerequisite missing.
        """
        # Prerequisite: the stage before first_stage must have a checkpoint.
        from ..configs.enums import STAGE_ORDER
        try:
            idx = STAGE_ORDER.index(first_stage)
        except ValueError:
            logger.warning("Unknown stage %s — skipping", first_stage)
            return False

        if idx > 0:
            prereq = STAGE_ORDER[idx - 1]
            if not await self.db.stage_exists(image_id, prereq):
                logger.warning(
                    "Skipping %s — no %s checkpoint for %s",
                    image_id, prereq.value, first_stage.value,
                )
                return False

        await self.db.add_work(first_stage.value, image_id)
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_images(
        self,
        image_paths: list[str],
        job_id: str,
    ) -> tuple[int, int]:
        """Submit image paths with Phase 2 smart routing.

        For each image:
          1. Apply force controls (force_rerun, force_stages, force_detect_models).
          2. Skip if all stages already complete (resume-safe).
          3. Register in image_meta.
          4. Queue work based on ``runtime.stages``:
             - If detect is first: queue per-model detect items
             - Otherwise: queue the first configured stage directly

        Args:
            image_paths: Absolute paths to image files.
            job_id: Logical batch / run identifier.

        Returns:
            ``(submitted, total_input)`` — *submitted* is new work queued,
            *total_input* is ``len(image_paths)``.
        """
        rt = self.config.runtime
        stages = rt.stages
        first_stage = stages[0] if stages else Stage.DETECT

        submitted = 0
        skipped = 0
        prerequisite_missing = 0

        for path in image_paths:
            image_id = Path(path).stem

            # Apply force controls (may clear cached data)
            await self._apply_force_controls(image_id)

            # Skip already-complete images (after force controls applied)
            if await self.db.all_stages_complete(image_id):
                skipped += 1
                continue

            # Register image in image_meta
            await self.db.register_image(image_id, str(path))

            # Queue work based on first stage
            if first_stage == Stage.DETECT:
                await self._queue_detect_work(image_id)
            else:
                queued = await self._queue_first_stage(image_id, first_stage)
                if not queued:
                    prerequisite_missing += 1
                    continue

            submitted += 1

        if skipped:
            logger.info(
                "Skipped %d already-completed image(s) for job '%s'",
                skipped, job_id,
            )
        if prerequisite_missing:
            logger.warning(
                "%d image(s) skipped — missing prerequisite stage checkpoint",
                prerequisite_missing,
            )
        logger.info(
            "Submitted %d image(s) for job '%s' (%d total, %d skipped)",
            submitted, job_id, len(image_paths), skipped,
        )
        return submitted, len(image_paths)

    async def submit_directory(
        self,
        image_dir: str | Path,
        job_id: str,
        extensions: frozenset[str] | tuple[str, ...] | None = None,
        recursive: bool = False,
    ) -> tuple[int, int]:
        """Discover and submit all images from a directory.

        Args:
            image_dir: Root directory to scan.
            job_id: Logical batch / run identifier.
            extensions: Lowercase file extensions to include. Defaults to _IMAGE_EXTENSIONS.
            recursive: If True, scan sub-directories (uses rglob).

        Returns:
            ``(submitted, total_input)`` — see :meth:`submit_images`.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        exts: frozenset[str] = (
            frozenset(e.lower() for e in extensions)
            if extensions is not None
            else self._IMAGE_EXTENSIONS
        )

        glob_fn = image_dir.rglob if recursive else image_dir.glob
        paths = sorted(
            p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in exts
        )

        submitted, total_input = await self.submit_images(
            [str(p) for p in paths], job_id
        )
        logger.info(
            "submit_directory: found %d image(s) in '%s'",
            len(paths), image_dir,
        )
        return submitted, total_input
