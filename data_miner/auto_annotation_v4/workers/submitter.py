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
from ..configs.enums import STAGE_ORDER, DetectorName, Stage
from ..configs.loader import compute_config_hash
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

    # Chunk sizes tuned for 1M-image scale (target 500-2000 rows per
    # SQLite transaction to amortize fsync cost).
    _REGISTER_CHUNK: int = 1000
    _WORK_CHUNK: int = 1000
    # Emit a progress log every N images during submit_images.
    _PROGRESS_EVERY: int = 10_000

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

    async def _plan_detect_work(
        self,
        image_id: str,
        pending: dict[str, list[str]],
    ) -> None:
        """Plan per-model detect work items for one image into *pending*.

        For each target model:
        - Skip if proposal already cached (unless forced)
        - Append image_id to ``pending["detect:{model.value}"]``

        After planning, check if the barrier is already met (all proposals
        cached from a previous run) and append to ``pending["detect:merge"]``.
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
            pending.setdefault(stage_key, []).append(image_id)
            models_queued += 1

        # If all proposals already cached, queue merge directly
        if models_queued == 0:
            model_values = [m.value for m in target_models]
            if await self.db.barrier_ready(image_id, model_values):
                # Check if detect stage result exists — if it does, no need to merge
                if not await self.db.stage_exists(image_id, Stage.DETECT):
                    pending.setdefault("detect:merge", []).append(image_id)
                    logger.debug(
                        "All proposals cached for %s — queued detect:merge",
                        image_id,
                    )

    # ------------------------------------------------------------------
    # Internal: queue non-detect first stage
    # ------------------------------------------------------------------

    async def _plan_first_stage(
        self,
        image_id: str,
        first_stage: Stage,
        pending: dict[str, list[str]],
    ) -> bool:
        """Plan work for a non-detect first stage into *pending*.

        Checks that the prerequisite stage exists before planning.

        Returns:
            True if work was planned, False if prerequisite missing.
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

        pending.setdefault(first_stage.value, []).append(image_id)
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

        current_hash = compute_config_hash(self.config, self.config.prompts_dir)

        # When force_stages is non-empty, clear_downstream wipes the cleared
        # stage (and everything after) for every input image. Enqueue must
        # target the earliest cleared stage rather than runtime.stages[0],
        # otherwise detect-first routing short-circuits on cached proposals
        # and nothing gets queued for the actual force-cleared stage.
        if rt.force_stages:
            effective_first = min(rt.force_stages, key=STAGE_ORDER.index)
        else:
            effective_first = first_stage

        submitted = 0
        skipped = 0
        prerequisite_missing = 0
        total = len(image_paths)

        # Buffers flushed in chunks to keep SQLite transactions large.
        register_buf: list[tuple[str, str]] = []
        work_buf: dict[str, list[str]] = {}

        async def _flush_register() -> None:
            if register_buf:
                await self.db.register_image_batch(register_buf)
                register_buf.clear()

        async def _flush_work(stage_key: str) -> None:
            ids = work_buf.get(stage_key)
            if ids:
                await self.db.add_work_batch(stage_key, ids)
                ids.clear()

        async def _flush_all_work() -> None:
            for key in list(work_buf.keys()):
                await _flush_work(key)

        for idx, path in enumerate(image_paths, 1):
            image_id = Path(path).stem

            # Apply force controls (may clear cached data)
            await self._apply_force_controls(image_id)

            # Check for config-hash mismatch on any enabled stage. A mismatch
            # triggers clear_downstream as a side-effect of should_run_stage,
            # which is the desired invalidation. If any stage was invalidated,
            # do NOT skip this image even if all_stages_complete still reports
            # True from a stale image_meta.status row.
            invalidated = False
            for stg in stages:
                if await self.db.should_run_stage(image_id, stg, current_hash):
                    invalidated = True
                    break

            # Skip already-complete images (after force controls applied)
            if not invalidated and await self.db.all_stages_complete(image_id):
                skipped += 1
            else:
                # Register image in image_meta (batched)
                register_buf.append((image_id, str(path)))
                if len(register_buf) >= self._REGISTER_CHUNK:
                    await _flush_register()

                # Plan work based on effective first stage (accounts for
                # force_stages clearing downstream of a non-detect stage).
                if effective_first == Stage.DETECT:
                    await self._plan_detect_work(image_id, work_buf)
                else:
                    queued = await self._plan_first_stage(
                        image_id, effective_first, work_buf
                    )
                    if not queued:
                        prerequisite_missing += 1
                        continue

                submitted += 1

                # Flush any work bucket that grew past the chunk threshold
                for key, ids in work_buf.items():
                    if len(ids) >= self._WORK_CHUNK:
                        await _flush_work(key)

            # Progress log every _PROGRESS_EVERY images
            if idx % self._PROGRESS_EVERY == 0:
                logger.info(
                    "submit_images progress: %d/%d scanned"
                    " (submitted=%d, skipped=%d) for job '%s'",
                    idx, total, submitted, skipped, job_id,
                )

        # Final flush for any partial buffers
        await _flush_register()
        await _flush_all_work()

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
        logger.info(
            "submit_directory: scanning '%s' (recursive=%s)",
            image_dir, recursive,
        )
        paths: list[Path] = []
        for p in glob_fn("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
                if len(paths) % self._PROGRESS_EVERY == 0:
                    logger.info(
                        "submit_directory: scanned %d image(s) so far",
                        len(paths),
                    )
        paths.sort()
        logger.info(
            "submit_directory: found %d image(s) in '%s'",
            len(paths), image_dir,
        )

        submitted, total_input = await self.submit_images(
            [str(p) for p in paths], job_id
        )
        return submitted, total_input
