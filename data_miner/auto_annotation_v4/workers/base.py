"""StageWorker — abstract base class for all pipeline stage workers.

Each concrete worker subclass:

1. Declares the ``stage`` class attribute (``"detect"``, ``"evaluate"``, or a
   compound stage like ``"detect:grounding_dino"`` for Phase 2).
2. Implements :meth:`process` to transform a :class:`~..configs.StageMessage`
   into a stage result (a Pydantic model).
3. Implements :meth:`_resolve_next_stage` to decide where work goes next
   based on the processing result.

The base :meth:`run` loop handles claiming from the SQLite work queue,
checkpoint resume/skip logic, atomic save-and-forward, retry counting,
dead-letter forwarding via :class:`~..checkpoint.CheckpointDB`, and
graceful shutdown on ``CancelledError``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TypeVar

import aiohttp
from pydantic import BaseModel

from ..checkpoint import CheckpointDB
from ..configs import (
    STAGE_ORDER,
    AutoAnnotationV4Config,
    Stage,
    StageMessage,
    compute_config_hash,
)
from ..output import OutputWriter

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("data_miner.auto_annotation_v4.worker")


# ---------------------------------------------------------------------------
# StageWorker
# ---------------------------------------------------------------------------


class StageWorker(ABC):
    """Base class for all pipeline stage workers.

    Handles work claiming from SQLite, checkpoint skip/resume logic,
    atomic save+forward, retry/failure, and graceful shutdown.

    Subclasses implement:
      - :meth:`process` — stage-specific logic (detection, evaluation, etc.)
      - :meth:`_resolve_next_stage` — routing decision after processing

    Parameters
    ----------
    config:
        Full pipeline configuration.
    db:
        Connected :class:`~..checkpoint.CheckpointDB` instance that serves as
        both the checkpoint store and the work queue (replaces Redis Streams
        from v3).
    output_writer:
        Optional :class:`~..output.OutputWriter` for stages that produce
        final output (e.g. finalize).
    worker_id:
        Unique name for this worker within its stage.  Defaults to
        ``"<stage>-<object id>"``.
    job_id:
        Optional job identifier for logging context.
    """

    stage: str = ""  # str not Stage — Phase 2 uses compound stages like "detect:grounding_dino"
    max_retries: int = 3
    needs_session: bool = True  # override to False for workers that make no HTTP calls

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        db: CheckpointDB,
        *,
        output_writer: OutputWriter | None = None,
        server_semaphore: asyncio.Semaphore | None = None,
        worker_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        if not self.stage:
            raise ValueError(
                f"{type(self).__name__} must define a non-empty 'stage' class attribute."
            )
        self.config = config
        self.db = db
        self.output_writer = output_writer
        self._server_semaphore = server_semaphore
        self.job_id = job_id
        self.worker_id = worker_id or f"{self.stage}-{id(self)}"
        self.logger = logging.getLogger(
            f"data_miner.auto_annotation_v4.{self.stage}.{self.worker_id}"
        )
        # Pre-compute config+prompt hash once per worker lifetime so every
        # checkpoint comparison uses a consistent fingerprint.
        self._config_hash: str = compute_config_hash(config, config.prompts_dir)
        self._running = False
        # Session created in run(), shared across all iterations.
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def process(self, msg: StageMessage) -> BaseModel:
        """Process one image.  Return a stage result model.

        The base class handles claiming, skip checks, save+forward, and retry.
        Subclass only implements the stage-specific processing logic.

        Parameters
        ----------
        msg:
            The incoming :class:`~..configs.StageMessage` containing the
            image_id, image_path, and stage metadata.

        Returns
        -------
        BaseModel
            The stage result (``DetectResult``, ``EvaluateResult``, etc.).
            Must be JSON-serialisable via ``model_dump_json()``.
        """
        ...

    @abstractmethod
    def _resolve_next_stage(self, result: BaseModel) -> Stage | str:
        """Determine next stage based on the processing result.

        Called by the base class after :meth:`process` succeeds.

        Parameters
        ----------
        result:
            The Pydantic model returned by :meth:`process`.

        Returns
        -------
        Stage | str
            A :class:`Stage` enum member or a compound stage string.
            ``Stage.DONE`` means terminal — no further work is queued.
        """
        ...

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main worker loop: claim -> skip-check -> process -> save_and_forward.

        Runs until :meth:`stop` is called or the task is cancelled.
        Each iteration attempts to claim one work item from the SQLite
        queue for this worker's stage.  If no work is available the loop
        sleeps briefly before retrying.

        Creates a single :class:`aiohttp.ClientSession` that lives for the
        worker's entire lifetime, shared across all iterations (avoids
        per-call session overhead).

        The loop is structured as two nested try/except layers:

        - **Inner try**: catches processing errors for a single image and
          records them via :meth:`~..checkpoint.CheckpointDB.fail_work`
          (which handles retry counting and dead-lettering).
        - **Inner CancelledError handler**: releases the claimed work item
          back to the queue so another worker (or a restart) can pick it up,
          then re-raises.  CancelledError is handled separately from
          Exception because it signals intentional shutdown, NOT a processing
          failure — we must never dead-letter an image just because the
          worker was stopped.
        - **Outer try**: catches the re-raised CancelledError to log a clean
          shutdown message and exit the loop.
        """
        self._running = True
        self.logger.info(
            "Worker %s started on stage '%s'", self.worker_id, self.stage
        )

        # One session per worker lifetime — avoids connector pool churn.
        if self.needs_session:
            self._session = aiohttp.ClientSession()
        try:
            while self._running:
                # 1. Claim work from queue
                image_id = await self.db.claim_work(self.stage, self.worker_id)
                if image_id is None:
                    await asyncio.sleep(1)
                    continue

                try:
                    # 2. Resume/skip check
                    #    For compound stages (e.g. "detect:grounding_dino"),
                    #    _checkpoint_stage() returns the parent Stage enum
                    #    (Stage.DETECT) so the stages table uses canonical keys.
                    stage_for_checkpoint = self._checkpoint_stage()
                    if await self.db.stage_exists(image_id, stage_for_checkpoint):
                        if not await self.db.should_run_stage(
                            image_id, stage_for_checkpoint, self._config_hash
                        ):
                            self.logger.info(
                                "Skipping %s/%s — checkpoint current",
                                image_id,
                                self.stage,
                            )
                            # Forward to next stage and mark current work done.
                            next_stage = self._next_stage_name()
                            await self.db.complete_work(image_id, self.stage)
                            if next_stage != Stage.DONE.value:
                                await self.db.add_work(next_stage, image_id)
                            continue

                    # 3. Resolve image path from image_meta
                    image_path = await self.db.resolve_image_path(image_id)

                    # 4. Build StageMessage
                    msg = StageMessage(
                        image_id=image_id,
                        image_path=image_path,
                        job_id=self.job_id or "",
                        stage=stage_for_checkpoint,
                    )

                    # 5. Process
                    t0 = time.perf_counter()
                    result = await self.process(msg)
                    timing_ms = (time.perf_counter() - t0) * 1000

                    # 6. Atomic save + forward
                    next_stage = self._resolve_next_stage(result)
                    next_stage_str = (
                        next_stage.value
                        if isinstance(next_stage, Stage)
                        else str(next_stage)
                    )
                    await self.db.save_and_forward(
                        image_id=image_id,
                        stage=stage_for_checkpoint,
                        data=result,
                        config_hash=self._config_hash,
                        next_stage=next_stage_str,
                        timing_ms=timing_ms,
                        # Pass the actual work_queue stage key so compound
                        # stages (e.g. "detect:merge") mark the correct row.
                        work_stage=self.stage,
                    )

                    self.logger.debug(
                        "Completed %s/%s -> %s (%.0fms)",
                        image_id,
                        self.stage,
                        next_stage_str,
                        timing_ms,
                    )

                except asyncio.CancelledError:
                    # Graceful shutdown — release work so next run can claim it.
                    # ALWAYS re-raise CancelledError: swallowing it would
                    # prevent asyncio from shutting down this task cleanly.
                    await self.db.release_work(image_id, self.stage)
                    raise

                except Exception as exc:
                    self.logger.exception("Failed %s/%s", image_id, self.stage)
                    await self.db.fail_work(image_id, self.stage, str(exc))

        except asyncio.CancelledError:
            self.logger.info("%s shutting down", self.worker_id)
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

        self.logger.info("Worker %s stopped", self.worker_id)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the run loop to exit after the current iteration.

        This is a cooperative stop — the worker finishes any in-progress
        image before exiting.  For immediate cancellation, cancel the
        asyncio task directly.
        """
        self._running = False

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _checkpoint_stage(self) -> Stage:
        """Return the :class:`Stage` enum for checkpoint operations.

        For simple stages (``"detect"``, ``"evaluate"``, etc.), returns the
        Stage enum directly by value.  For compound stages
        (``"detect:grounding_dino"``), returns the parent stage
        (``Stage.DETECT``) by splitting on ``":"``.

        Override in subclasses that need non-standard checkpoint-stage mapping.
        """
        base = self.stage.split(":")[0]
        return Stage(base)

    def _next_stage_name(self) -> str:
        """Return the default next stage name for skip/forward.

        Uses :data:`STAGE_ORDER` to find the stage after
        :meth:`_checkpoint_stage`.  Returns ``"done"`` if the current stage
        is the last one in the pipeline.
        """
        current = self._checkpoint_stage()
        try:
            idx = STAGE_ORDER.index(current)
        except ValueError:
            return Stage.DONE.value
        next_idx = idx + 1
        if next_idx < len(STAGE_ORDER):
            return STAGE_ORDER[next_idx].value
        return Stage.DONE.value

    # ------------------------------------------------------------------
    # Checkpoint convenience wrappers
    # ------------------------------------------------------------------

    async def save_checkpoint(
        self,
        image_id: str,
        stage: Stage,
        data: BaseModel,
    ) -> None:
        """Save a stage checkpoint for *image_id*.

        Convenience wrapper around :meth:`CheckpointDB.save_stage` that
        automatically passes the worker's pre-computed config hash.

        Parameters
        ----------
        image_id:
            Target image identifier.
        stage:
            Pipeline stage this checkpoint belongs to.
        data:
            Pydantic model containing the stage result.
        """
        await self.db.save_stage(image_id, stage, data, self._config_hash)

    async def load_checkpoint(
        self,
        image_id: str,
        stage: Stage,
        model_class: type[T],
    ) -> T | None:
        """Load a stage checkpoint for *image_id*, or ``None`` if absent.

        Convenience wrapper around :meth:`CheckpointDB.load_stage`.

        Parameters
        ----------
        image_id:
            Target image identifier.
        stage:
            Pipeline stage to load from.
        model_class:
            Pydantic model class to deserialise the stored JSON into.

        Returns
        -------
        T | None
            Validated instance of *model_class*, or ``None`` if no checkpoint
            exists for the given image and stage.
        """
        return await self.db.load_stage(image_id, stage, model_class)
