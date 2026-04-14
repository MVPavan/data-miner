"""StageWorker — abstract base class for all pipeline stage workers.

Each concrete worker subclass:

1. Declares the ``stage`` class attribute (``"detect"``, ``"evaluate"``, or
   ``"refine"``).
2. Implements :meth:`process` to transform a :class:`~..contracts.StageMessage`
   into the next message (or ``None`` for terminal handling).

The base :meth:`run` loop handles reading from Redis Streams, checkpoint
resume/skip logic, retry counting, dead-letter forwarding, and graceful
shutdown on ``CancelledError``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..checkpoint import CheckpointManager
from ..config import AutoAnnotationV3Config, compute_config_hash
from ..contracts import StageMessage
from .messaging import RedisMessageBroker

if TYPE_CHECKING:
    pass  # keep for future type-only imports

logger = logging.getLogger("data_miner.auto_annotation_v3.worker")

# Canonical stage progression — must match STAGE_ORDER in checkpoint.py.
_PIPELINE_STAGES: list[str] = ["detect", "evaluate", "refine", "done"]


# ---------------------------------------------------------------------------
# StageWorker
# ---------------------------------------------------------------------------


class StageWorker(ABC):
    """Base class for all pipeline stage workers.

    Subclasses **must** set the ``stage`` class attribute and implement
    :meth:`process`.

    Parameters
    ----------
    config:
        Full pipeline configuration.
    broker:
        Connected :class:`~.messaging.RedisMessageBroker` instance.
    checkpoint_mgr:
        :class:`~..checkpoint.CheckpointManager` for resume/skip decisions.
    worker_id:
        Unique name for this worker within its consumer group.  Defaults to
        ``"<stage>-<object id>"``.
    """

    stage: str = ""  # override in each concrete subclass
    max_retries: int = 3

    def __init__(
        self,
        config: AutoAnnotationV3Config,
        broker: RedisMessageBroker,
        checkpoint_mgr: CheckpointManager,
        worker_id: str | None = None,
    ) -> None:
        if not self.stage:
            raise ValueError(
                f"{type(self).__name__} must define a non-empty 'stage' class attribute."
            )
        self.config = config
        self.broker = broker
        self.checkpoint = checkpoint_mgr
        self.worker_id = worker_id or f"{self.stage}-{id(self)}"
        self.logger = logging.getLogger(
            f"data_miner.auto_annotation_v3.{self.stage}.{self.worker_id}"
        )
        self._running = False
        # Pre-compute config+prompt hash once per worker lifetime.
        self._config_hash: str = compute_config_hash(config, config.prompts_dir)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def process(self, msg: StageMessage) -> StageMessage | None:
        """Process a single stage message.

        Parameters
        ----------
        msg:
            The incoming :class:`~..contracts.StageMessage`.

        Returns
        -------
        StageMessage | None
            The message to forward to the next stage, or ``None`` if this
            worker handles terminal disposition itself (e.g. writing to
            ``done`` internally).
        """
        ...

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Read–process–forward loop.  Runs until :meth:`stop` is called or
        the task is cancelled.
        """
        self._running = True
        self.logger.info(
            "Worker %s started on stage '%s'", self.worker_id, self.stage
        )

        while self._running:
            try:
                messages = await self.broker.read(
                    self.stage, self.worker_id, count=1, block_ms=5000
                )
                if not messages:
                    continue

                for msg_id, raw_msg in messages:
                    await self._handle_message(msg_id, raw_msg)

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.error(
                    "Unexpected error in worker loop", exc_info=True
                )
                await asyncio.sleep(1)

        self.logger.info("Worker %s stopped", self.worker_id)

    async def _handle_message(self, msg_id: str, raw_msg: dict) -> None:
        """Process one message with retry / dead-letter logic."""
        try:
            stage_msg = StageMessage(**raw_msg)
        except Exception:
            self.logger.error(
                "Malformed message %s — sending to dead-letter", msg_id, exc_info=True
            )
            await self.broker.send_to_dead_letter(
                self.stage, msg_id, raw_msg, "Malformed StageMessage"
            )
            return

        # ---- Resume / skip check ----------------------------------------
        if self.checkpoint.exists(stage_msg.image_id, self.stage):
            if not self.checkpoint.should_run_stage(
                stage_msg.image_id, self.stage, self._config_hash
            ):
                self.logger.info(
                    "Skipping %s/%s — already complete",
                    stage_msg.image_id,
                    self.stage,
                )
                await self._forward_and_ack(stage_msg, msg_id)
                return

        # ---- Process --------------------------------------------------------
        t0 = time.monotonic()
        try:
            result = await self.process(stage_msg)
            elapsed_ms = (time.monotonic() - t0) * 1000

            self.checkpoint.update_meta(
                stage_msg.image_id,
                self.stage,
                config_hash=self._config_hash,
                prompt_version="",
                timing_ms=elapsed_ms,
            )

            if result is not None:
                await self.broker.submit(result.stage, result.model_dump())

            await self.broker.ack(self.stage, msg_id)
            self.logger.info(
                "Processed %s in %.0f ms", stage_msg.image_id, elapsed_ms
            )

        except Exception as exc:
            self.logger.error(
                "Failed to process %s: %s", stage_msg.image_id, exc, exc_info=True
            )
            stage_msg = stage_msg.model_copy(update={"attempt": stage_msg.attempt + 1})

            if stage_msg.attempt >= self.max_retries:
                self.logger.error(
                    "Max retries reached for %s — sending to dead-letter",
                    stage_msg.image_id,
                )
                await self.broker.send_to_dead_letter(
                    self.stage, msg_id, raw_msg, str(exc)
                )
            else:
                # Re-queue for another attempt and ack the original delivery.
                await self.broker.submit(self.stage, stage_msg.model_dump())
                await self.broker.ack(self.stage, msg_id)

    async def _forward_and_ack(self, stage_msg: StageMessage, msg_id: str) -> None:
        """Forward *stage_msg* to the next stage and ack the current one."""
        next_stage = self._next_stage()
        next_msg = stage_msg.forward(next_stage)
        await self.broker.submit(next_msg.stage, next_msg.model_dump())
        await self.broker.ack(self.stage, msg_id)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the run loop to exit after the current message finishes."""
        self._running = False

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _next_stage(self) -> str:
        """Return the name of the stage that follows this worker's stage."""
        try:
            idx = _PIPELINE_STAGES.index(self.stage)
        except ValueError:
            return "done"
        next_idx = idx + 1
        return _PIPELINE_STAGES[next_idx] if next_idx < len(_PIPELINE_STAGES) else "done"

    # ------------------------------------------------------------------
    # Checkpoint convenience wrappers
    # ------------------------------------------------------------------

    def save_checkpoint(self, image_id: str, stage: str, data) -> None:
        """Save a stage checkpoint for *image_id*."""
        self.checkpoint.save(image_id, stage, data)

    def load_checkpoint(self, image_id: str, stage: str, model_class):
        """Load a stage checkpoint for *image_id*, or ``None`` if absent."""
        return self.checkpoint.load(image_id, stage, model_class)
