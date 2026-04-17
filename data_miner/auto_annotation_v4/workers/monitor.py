"""PipelineMonitor — real-time status reporting for the annotation pipeline.

Replaces v3's Redis stream introspection with direct SQLite queries against
:class:`~..checkpoint.CheckpointDB`.  Also runs periodic stale claim recovery
(replaces Redis TTL auto-expiry).

Usage::

    monitor = PipelineMonitor(db)
    await monitor.print_status()
    completed = await monitor.wait_for_completion(total_images=500)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..checkpoint import CheckpointDB
from ..configs.enums import Stage

logger = logging.getLogger("data_miner.auto_annotation_v4.monitor")

_ALL_STAGES: tuple[str, ...] = tuple(s.value for s in Stage)


class PipelineMonitor:
    """Monitors pipeline progress via SQL queries against CheckpointDB.

    Replaces v3's Redis stream introspection with direct SQLite queries.
    Also runs periodic stale claim recovery (replaces Redis TTL auto-expiry).

    Args:
        db: Connected CheckpointDB instance.
        lock_ttl: Seconds before stale processing claims are recovered.
        max_retries: Max attempts before moving to failures table.
    """

    def __init__(
        self,
        db: CheckpointDB,
        *,
        lock_ttl: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.db = db
        self.lock_ttl = lock_ttl
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    async def get_status(self) -> dict:
        """Return a progress snapshot from CheckpointDB.

        Returns:
            Dict with keys: ``proposals``, ``stages``, ``queue``,
            ``processing``, ``failed``.  The ``processing`` key is a
            convenience count of all work_queue items currently claimed.
        """
        summary = await self.db.progress_summary()

        # Compute a flat processing count across all stages for quick access.
        processing = 0
        for stage_counts in summary.get("queue", {}).values():
            processing += stage_counts.get("processing", 0)
        summary["processing"] = processing

        return summary

    async def print_status(self) -> None:
        """Print a formatted progress table to the logger."""
        status = await self.get_status()
        _print_status_table(status)

    # ------------------------------------------------------------------
    # Stale claim recovery
    # ------------------------------------------------------------------

    async def recover_stale(self) -> int:
        """Sweep for stale processing claims and recover them.

        Returns:
            Number of recovered items.
        """
        recovered = await self.db.recover_stale(self.lock_ttl, self.max_retries)
        if recovered:
            logger.info("Recovered %d stale claims", recovered)
        return recovered

    # ------------------------------------------------------------------
    # Completion helper
    # ------------------------------------------------------------------

    async def wait_for_completion(
        self,
        total_images: int,
        *,
        poll_interval: float = 5.0,
        timeout: float | None = None,
        print_progress: bool = True,
    ) -> bool:
        """Block until all images reach finalize or fail.

        Ground truth is image_meta status, not queue counts.
        Runs stale recovery on every poll cycle.

        Args:
            total_images:
                Total number of input images (use *total_input* from submitter).
            poll_interval:
                Seconds between status checks.
            timeout:
                Max seconds to wait (``None`` = unlimited).
            print_progress:
                Whether to print progress each cycle.

        Returns:
            ``True`` if all images completed before timeout.
        """
        start = time.monotonic()
        logger.info(
            "Waiting for %d images to complete (poll every %.1fs)...",
            total_images,
            poll_interval,
        )

        while True:
            # Ground truth: stage checkpoint counts + failures
            status = await self.get_status()
            done = status.get("stages", {}).get(Stage.FINALIZE.value, 0)
            failed = status.get("failed", 0)
            finished = done + failed

            if print_progress:
                elapsed = time.monotonic() - start
                self._print_progress(status, finished, total_images, elapsed)

            if finished >= total_images:
                logger.info(
                    "All %d images finished (%d complete, %d failed).",
                    total_images,
                    done,
                    failed,
                )
                return True

            # Stale recovery on every poll
            await self.recover_stale()

            if timeout is not None and (time.monotonic() - start) >= timeout:
                logger.warning(
                    "Timed out after %.0f s -- %d / %d images finished.",
                    timeout,
                    finished,
                    total_images,
                )
                return False

            await asyncio.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Checkpoint-based queries
    # ------------------------------------------------------------------

    async def completed_count(self) -> int:
        """Return the number of images that have reached the finalize stage."""
        status = await self.get_status()
        return status.get("stages", {}).get(Stage.FINALIZE.value, 0)

    async def failed_count(self) -> int:
        """Return the number of images in the failures table."""
        status = await self.get_status()
        return status.get("failed", 0)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _print_progress(
        self,
        status: dict,
        finished: int,
        total: int,
        elapsed: float,
    ) -> None:
        """Format and print a progress line."""
        pct = 100.0 * finished / total if total > 0 else 0.0

        stages = status.get("stages", {})
        queue = status.get("queue", {})
        failed = status.get("failed", 0)
        processing = status.get("processing", 0)

        # Per-stage checkpoint counts
        detect_done = stages.get(Stage.DETECT.value, 0)
        eval_done = stages.get(Stage.EVALUATE.value, 0)
        refine_done = stages.get(Stage.REFINE.value, 0)
        finalize_done = stages.get(Stage.FINALIZE.value, 0)

        # Per-stage pending queue counts
        detect_q = queue.get(Stage.DETECT.value, {}).get("pending", 0)
        eval_q = queue.get(Stage.EVALUATE.value, {}).get("pending", 0)
        refine_q = queue.get(Stage.REFINE.value, {}).get("pending", 0)
        finalize_q = queue.get(Stage.FINALIZE.value, {}).get("pending", 0)

        logger.info(
            "[%6.0fs] %d/%d (%.1f%%)  "
            "detect=%d(%d)  eval=%d(%d)  refine=%d(%d)  finalize=%d(%d)  "
            "processing=%d  failed=%d",
            elapsed,
            finished,
            total,
            pct,
            detect_done,
            detect_q,
            eval_done,
            eval_q,
            refine_done,
            refine_q,
            finalize_done,
            finalize_q,
            processing,
            failed,
        )


# ---------------------------------------------------------------------------
# Formatting helpers (module-level)
# ---------------------------------------------------------------------------


def _print_status_table(status: dict) -> None:
    """Print a formatted status table to the logger.

    Shows per-stage checkpoint counts, queue depth, and failure totals.
    """
    stages = status.get("stages", {})
    queue = status.get("queue", {})
    proposals = status.get("proposals", {})
    failed = status.get("failed", 0)

    col_w = 14
    header = f"{'Stage':<{col_w}} {'Done':>8} {'Pending':>8} {'Processing':>10}"
    sep = "-" * len(header)

    lines = [sep, header, sep]

    for stage in _ALL_STAGES:
        done = stages.get(stage, 0)
        q = queue.get(stage, {})
        pending = q.get("pending", 0)
        processing = q.get("processing", 0)
        lines.append(
            f"{stage:<{col_w}} {done:>8} {pending:>8} {processing:>10}"
        )

    lines.append(sep)

    if proposals:
        lines.append("Proposals:")
        for model, count in sorted(proposals.items()):
            lines.append(f"  {model}: {count}")
        lines.append(sep)

    lines.append(f"Failed: {failed}")
    lines.append(sep)

    logger.info("\n%s", "\n".join(lines))
