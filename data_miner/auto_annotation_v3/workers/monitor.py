"""PipelineMonitor — real-time status reporting for the annotation pipeline.

Provides queue-depth / pending-count snapshots and a blocking
:meth:`~PipelineMonitor.wait_for_completion` helper.

Usage::

    monitor = PipelineMonitor(broker, checkpoint_mgr)
    await monitor.print_status()
    await monitor.wait_for_completion(total_images=500)
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..checkpoint import CheckpointManager
from .messaging import RedisMessageBroker

logger = logging.getLogger("data_miner.auto_annotation_v3.monitor")

_ALL_STAGES: tuple[str, ...] = (
    "detect",
    "evaluate",
    "refine",
    "done",
    "dead_letter",
)


class PipelineMonitor:
    """Monitors pipeline progress via Redis stream introspection.

    Parameters
    ----------
    broker:
        A connected :class:`~.messaging.RedisMessageBroker` instance.
    checkpoint_mgr:
        :class:`~..checkpoint.CheckpointManager` used for completion queries.
    """

    def __init__(
        self,
        broker: RedisMessageBroker,
        checkpoint_mgr: CheckpointManager,
    ) -> None:
        self.broker = broker
        self.checkpoint = checkpoint_mgr

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, dict[str, int]]:
        """Return a snapshot of stream lengths and pending counts.

        Returns
        -------
        dict
            ``{stage: {"queue_length": int, "pending": int}}`` for each stage.
        """
        status: dict[str, dict[str, int]] = {}
        for stage in _ALL_STAGES:
            queue_length = await self.broker.stream_length(stage)
            pending = await self.broker.pending_count(stage)
            status[stage] = {"queue_length": queue_length, "pending": pending}
        return status

    async def print_status(self) -> None:
        """Print a formatted status table to stdout."""
        status = await self.get_status()
        _print_status_table(status)

    # ------------------------------------------------------------------
    # Completion helper
    # ------------------------------------------------------------------

    async def wait_for_completion(
        self,
        total_images: int,
        poll_interval: float = 5.0,
        timeout: float | None = None,
        print_progress: bool = True,
    ) -> bool:
        """Block until *total_images* reach the ``done`` stream (or timeout).

        Parameters
        ----------
        total_images:
            Expected total number of images in this job.
        poll_interval:
            Seconds between Redis polls.
        timeout:
            Maximum seconds to wait.  ``None`` means wait indefinitely.
        print_progress:
            If ``True``, print a status line each poll cycle.

        Returns
        -------
        bool
            ``True`` if all images reached ``done`` before *timeout*,
            ``False`` otherwise.
        """
        start = time.monotonic()
        logger.info(
            "Waiting for %d images to complete (poll every %.1fs)…",
            total_images,
            poll_interval,
        )

        while True:
            status = await self.get_status()
            done_count = status["done"]["queue_length"]
            dead_count = status["dead_letter"]["queue_length"]
            finished = done_count + dead_count

            if print_progress:
                elapsed = time.monotonic() - start
                _print_progress(status, finished, total_images, elapsed)

            if finished >= total_images:
                logger.info(
                    "All %d images finished (%d done, %d dead-letter).",
                    total_images,
                    done_count,
                    dead_count,
                )
                return True

            if timeout is not None and (time.monotonic() - start) >= timeout:
                logger.warning(
                    "Timed out after %.0f s — %d / %d images finished.",
                    timeout,
                    finished,
                    total_images,
                )
                return False

            await asyncio.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Checkpoint-based queries
    # ------------------------------------------------------------------

    def completed_image_ids(self) -> list[str]:
        """Return image IDs whose all pipeline stages are complete."""
        return [
            iid
            for iid in self.checkpoint.image_ids()
            if self.checkpoint.all_stages_complete(iid)
        ]

    def completion_count(self) -> int:
        """Return the number of fully-complete images (checkpoint-based)."""
        return len(self.completed_image_ids())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_status_table(status: dict[str, dict[str, int]]) -> None:
    col_w = 14
    header = f"{'Stage':<{col_w}} {'Queue':>8} {'Pending':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for stage, info in status.items():
        print(
            f"{stage:<{col_w}} {info['queue_length']:>8} {info['pending']:>8}"
        )
    print(sep)


def _print_progress(
    status: dict[str, dict[str, int]],
    finished: int,
    total: int,
    elapsed: float,
) -> None:
    pct = 100.0 * finished / total if total > 0 else 0.0
    done = status["done"]["queue_length"]
    dead = status["dead_letter"]["queue_length"]
    detect_q = status["detect"]["queue_length"]
    eval_q = status["evaluate"]["queue_length"]
    refine_q = status["refine"]["queue_length"]
    print(
        f"[{elapsed:6.0f}s] {finished}/{total} ({pct:.1f}%)  "
        f"detect={detect_q}  eval={eval_q}  refine={refine_q}  "
        f"done={done}  dead={dead}"
    )
