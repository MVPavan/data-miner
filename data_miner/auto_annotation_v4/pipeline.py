"""Pipeline orchestrator for auto_annotation_v4.

Launches stage workers, manages the pipeline lifecycle, and coordinates
between the SQLite-backed CheckpointDB (work queue + checkpoints), the
JobSubmitter, and the PipelineMonitor.

Key differences from v3:
- No Redis: all state lives in a single per-job SQLite database.
- Config continuity check warns when the config/prompt hash changes
  between runs so stale stages can be invalidated.
- Signal handling: SIGINT + SIGTERM trigger graceful shutdown of workers
  and the monitor task.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from .checkpoint import CheckpointDB
from .configs.enums import DetectorName, Stage
from .configs.loader import compute_config_hash, load_config
from .configs.settings import AutoAnnotationV4Config
from .output import OutputWriter
from .workers.submitter import JobSubmitter
from .workers.monitor import PipelineMonitor
from .utils import configure_logging, get_logger

logger = get_logger("pipeline")


class AutoAnnotationPipelineV4:
    """Manages the full v4 annotation pipeline.

    Orchestrates the pipeline lifecycle:
      1. Preflight health checks (detector servers + VLM).
      2. Database connection + job info persistence.
      3. Config continuity check (warns if config/prompts changed).
      4. Image submission to the detect work queue.
      5. Worker creation and async task launching.
      6. Progress monitoring + stale claim recovery.
      7. Graceful shutdown on completion, timeout, or signal.

    All state is stored in a single per-job SQLite database
    (``pipeline.db``), eliminating the Redis dependency from v3.

    Parameters
    ----------
    config:
        Validated pipeline configuration.
    job_id:
        Optional explicit job identifier.  Falls back to
        ``config.runtime.job_id``, then to a timestamp-based default.
    """

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        job_id: str | None = None,
    ) -> None:
        self.config = config

        # Prefer explicit job_id, then config.runtime.job_id, then autogen.
        self.job_id = (
            job_id
            or config.runtime.job_id
            or f"job_{datetime.now():%Y%m%d_%H%M%S}"
        )

        # Setup output directory
        self.job_dir = Path(config.output.job_dir) / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # Save frozen config as JSON (config.yaml extension kept for readability)
        (self.job_dir / "config.yaml").write_text(
            config.model_dump_json(indent=2)
        )

        # ONE database for everything: checkpoints, work queue, proposals, metadata.
        db_path = self.job_dir / config.database.filename
        self.db = CheckpointDB(
            db_path,
            lock_ttl=config.database.lock_ttl,
            max_retries=config.database.max_retries,
        )
        self.output_writer = OutputWriter(self.job_dir)
        self.submitter = JobSubmitter(config, self.db)
        self.monitor = PipelineMonitor(
            self.db,
            lock_ttl=config.database.lock_ttl,
            max_retries=config.database.max_retries,
        )

        self._workers: list = []
        self._tasks: list[asyncio.Task] = []
        self._monitor_task: asyncio.Task | None = None
        self._shutdown_requested = False

    # ------------------------------------------------------------------
    # Config continuity check
    # ------------------------------------------------------------------

    async def _check_config_continuity(self) -> None:
        """Warn if the config/prompt hash has changed since the last run.

        Compares the stored config_hash in the job_info table against the
        current hash.  If they differ, the user is warned that stale
        downstream stages will be invalidated on a per-image basis by
        :meth:`CheckpointDB.should_run_stage`.

        In interactive mode (stdin is a TTY), prompts for confirmation.
        In non-interactive mode, proceeds automatically with a log message.
        """
        stored = await self.db.get_job_info()
        if stored is None:
            return  # fresh job

        current_hash = compute_config_hash(self.config, self.config.prompts_dir)
        if stored["config_hash"] == current_hash:
            return  # same config

        logger.warning(
            "Config changed since last run.\n  Stored: %s\n  Current: %s\n"
            "Stale stages will be invalidated.",
            stored["config_hash"],
            current_hash,
        )
        if sys.stdin.isatty():
            response = input("Continue? [y/N]: ").strip().lower()
            if response != "y":
                logger.info("Aborted by user.")
                sys.exit(0)
        else:
            logger.info("Non-interactive — proceeding with invalidation.")

    # ------------------------------------------------------------------
    # Preflight health checks
    # ------------------------------------------------------------------

    def _check_endpoint(self, url: str, timeout: float = 3.0) -> bool:
        """Return True if *url* responds with HTTP 200.

        Parameters
        ----------
        url:
            Full URL to probe (typically a ``/health`` endpoint).
        timeout:
            Connection timeout in seconds.
        """
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def preflight(self) -> None:
        """Verify all required servers are online before the pipeline starts.

        Checks:
        - Every enabled detector server (``/health`` endpoint).
        - VLM server (``/health`` endpoint).

        Raises
        ------
        RuntimeError
            If any server is unreachable, with a summary of all failures.
        """
        logger.info("Running preflight health checks ...")
        failures: list[str] = []

        # 1. Detector servers
        for name, cfg in self.config.servers.enabled_detectors().items():
            url = f"http://localhost:{cfg.port}/health"
            if self._check_endpoint(url):
                logger.info("  + %s (port %d)", name.value, cfg.port)
            else:
                msg = f"Detector '{name.value}' not reachable at localhost:{cfg.port}/health"
                logger.error("  x %s", msg)
                failures.append(msg)

        # 2. VLM server
        vlm_base = self.config.servers.vlm.url.rstrip("/")
        # vLLM exposes /health at the base URL (not under /v1)
        vlm_health = vlm_base.replace("/v1", "") + "/health"
        if self._check_endpoint(vlm_health):
            logger.info("  + VLM (%s)", vlm_health)
        else:
            msg = f"VLM not reachable at {vlm_health}"
            logger.error("  x %s", msg)
            failures.append(msg)

        if failures:
            summary = "\n  - ".join(failures)
            raise RuntimeError(
                f"Preflight failed — {len(failures)} server(s) unreachable:\n  - {summary}\n"
                "Start the missing servers and retry."
            )
        logger.info("Preflight passed — all servers healthy.")

    # ------------------------------------------------------------------
    # Worker creation
    # ------------------------------------------------------------------

    def _create_workers(self) -> list:
        """Create worker instances based on config.

        Phase 2: per-model detect workers + merge workers replace the
        monolithic DetectWorker.  Workers are only created for stages
        listed in ``runtime.stages``.

        Returns
        -------
        list
            All worker instances across all active stages.
        """
        from .stages.detect import DetectMergeWorker
        from .stages.detect_model import DetectModelWorker
        from .stages.evaluate import EvaluateWorker
        from .stages.refine import RefineWorker
        from .stages.finalize import FinalizeWorker

        active_stages = set(self.config.runtime.stages)
        workers = []

        # ── Per-server semaphores (limit concurrent requests) ─────────
        self._server_semaphores: dict[str, asyncio.Semaphore] = {}
        for name, cfg in self.config.servers.enabled_detectors().items():
            self._server_semaphores[name.value] = asyncio.Semaphore(cfg.max_batch_size)

        # ── Detect: per-model workers + merge workers ─────────────────
        if Stage.DETECT in active_stages:
            # Determine target models (runtime.detect_models or all enabled)
            enabled = self.config.servers.enabled_detectors()
            if self.config.runtime.detect_models:
                target_models = [
                    m for m in self.config.runtime.detect_models if m in enabled
                ]
            else:
                target_models = list(enabled.keys())

            # Per-model detect workers — each gets its model's semaphore
            for model_name in target_models:
                sem = self._server_semaphores.get(model_name.value)
                for i in range(self.config.workers.detect_per_model):
                    w = DetectModelWorker(
                        config=self.config,
                        db=self.db,
                        model_name=model_name,
                        enabled_models=target_models,
                        server_semaphore=sem,
                        worker_id=f"detect:{model_name.value}-{i}",
                        job_id=self.job_id,
                    )
                    workers.append(w)

            # Merge workers (no semaphore — no HTTP calls)
            for i in range(self.config.workers.detect_merge):
                w = DetectMergeWorker(
                    config=self.config,
                    db=self.db,
                    output_writer=self.output_writer,
                    worker_id=f"detect:merge-{i}",
                    job_id=self.job_id,
                )
                workers.append(w)

        # ── Evaluate ──────────────────────────────────────────────────
        if Stage.EVALUATE in active_stages:
            for i in range(self.config.workers.evaluate_count):
                w = EvaluateWorker(
                    config=self.config,
                    db=self.db,
                    worker_id=f"evaluate-{i}",
                    job_id=self.job_id,
                )
                workers.append(w)

        # ── Refine — gets SAM server semaphore ────────────────────────
        if Stage.REFINE in active_stages:
            # Find the SAM semaphore (prefer sam3_dart, fallback to sam3)
            sam_sem = None
            for sam_name in (DetectorName.SAM3_DART, DetectorName.SAM3):
                if sam_name.value in self._server_semaphores:
                    sam_sem = self._server_semaphores[sam_name.value]
                    break

            for i in range(self.config.workers.refine_count):
                w = RefineWorker(
                    config=self.config,
                    db=self.db,
                    server_semaphore=sam_sem,
                    worker_id=f"refine-{i}",
                    job_id=self.job_id,
                )
                workers.append(w)

        # ── Finalize ──────────────────────────────────────────────────
        if Stage.FINALIZE in active_stages:
            for i in range(self.config.workers.finalize_count):
                w = FinalizeWorker(
                    config=self.config,
                    db=self.db,
                    output_writer=self.output_writer,
                    worker_id=f"finalize-{i}",
                    job_id=self.job_id,
                )
                workers.append(w)

        return workers

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _request_shutdown(self) -> None:
        """Signal handler — initiate two-phase graceful shutdown.

        Phase 1 (cooperative): tell workers to stop after their current image.
        Phase 2 (forced, after 30s): cancel tasks that didn't exit cooperatively.

        A second signal forces immediate cancellation.
        """
        if getattr(self, "_shutdown_requested", False):
            # Second signal — force immediately.
            logger.warning("Second signal received — forcing shutdown")
            self._force_shutdown()
            return

        logger.info("Shutting down gracefully (30s deadline)...")
        self._shutdown_requested = True

        # Phase 1: cooperative stop — workers finish current image.
        for w in self._workers:
            w.stop()
        # Cancel monitor so run() can proceed to cleanup.
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        # Schedule forced shutdown after 30s.
        asyncio.get_running_loop().call_later(30.0, self._force_shutdown)

    def _force_shutdown(self) -> None:
        """Force-cancel any worker tasks that didn't stop cooperatively."""
        for t in self._tasks:
            if not t.done():
                t.cancel()

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    async def run(
        self,
        image_paths: list[str] | None = None,
        image_dir: str | None = None,
    ) -> None:
        """Run the full pipeline.

        Execution flow:
          1. Preflight health checks (detector servers + VLM).
          2. Connect to the SQLite database.
          3. Save job info (job_id, image_dir, config hash, prompt version).
          4. Check config continuity (warn if config changed since last run).
          5. Submit images to the detect work queue.
          6. Create workers and launch async tasks.
          7. Monitor progress until all images complete or fail.
          8. Write summary and shut down.

        Input sources are resolved in order of priority:
          1. ``image_paths`` argument
          2. ``image_dir`` argument
          3. ``config.runtime.image_paths``
          4. ``config.runtime.image_dir``

        Parameters
        ----------
        image_paths:
            Explicit list of image file paths to process.
        image_dir:
            Directory to scan for image files.
        """
        # Preflight: verify all servers are reachable before committing.
        self.preflight()

        # Connect to the SQLite database.
        await self.db.connect()

        try:
            # Compute config + prompt hash for this run.
            config_hash = compute_config_hash(self.config, self.config.prompts_dir)

            # Save job-level metadata.
            await self.db.save_job_info(
                job_id=self.job_id,
                image_dir=image_dir or self.config.runtime.image_dir,
                config_hash=config_hash,
                prompt_version=config_hash,  # prompt version is embedded in the hash
            )

            # Check config continuity (warns if changed, optionally aborts).
            await self._check_config_continuity()

            # Resolve inputs: arg -> config.runtime fallback
            image_dir = image_dir or self.config.runtime.image_dir
            image_paths = image_paths or list(self.config.runtime.image_paths)

            if image_paths:
                submitted, total_input = await self.submitter.submit_images(
                    image_paths, self.job_id
                )
            elif image_dir:
                submitted, total_input = await self.submitter.submit_directory(
                    image_dir, self.job_id
                )
            else:
                raise ValueError(
                    "No input provided — set runtime.image_dir or "
                    "runtime.image_paths in config (or pass image_dir/"
                    "image_paths to run())"
                )

            if submitted == 0:
                logger.info(
                    "All images already complete for job %s — nothing to do",
                    self.job_id,
                )
                return

            logger.info(
                "Submitted %d/%d images for job %s",
                submitted,
                total_input,
                self.job_id,
            )

            # Write classes file.
            class_map = {name: cfg.id for name, cfg in self.config.classes.items()}
            self.output_writer.write_classes_file(class_map)

            # Launch workers.
            self._workers = self._create_workers()
            self._tasks = [asyncio.create_task(w.run()) for w in self._workers]

            # Setup signal handling for graceful shutdown.
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)

            # Wait for completion — use total_input (not submitted) so
            # same-job-id reruns wait for ALL images, not just new ones.
            # Wrapped in a task so _shutdown() can cancel it on Ctrl+C.
            self._monitor_task = asyncio.create_task(
                self.monitor.wait_for_completion(
                    total_input,
                    poll_interval=5.0,
                    timeout=None,
                    print_progress=True,
                )
            )
            try:
                completed = await self._monitor_task
            except asyncio.CancelledError:
                logger.info("Monitor cancelled — shutting down")
                completed = False

            # Wait for workers to finish (cooperative shutdown gives them time
            # to complete their current image and release claimed work).
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            if completed:
                logger.info("Pipeline complete for job %s", self.job_id)
            else:
                logger.warning("Pipeline did not complete for job %s", self.job_id)

            # Write summary.
            status = await self.monitor.get_status()
            self.output_writer.write_summary(
                {
                    "job_id": self.job_id,
                    "total_images": total_input,
                    "submitted": submitted,
                    "status": status,
                    "completed": completed,
                }
            )

        finally:
            self._force_shutdown()
            await self.db.close()

    async def run_single_image(self, image_path: str) -> None:
        """Run pipeline for a single image (convenience method).

        Parameters
        ----------
        image_path:
            Absolute path to the image file.
        """
        await self.run(image_paths=[image_path])


def run_pipeline(
    user_config: str | None = None,
    overrides: list[str] | None = None,
) -> None:
    """Synchronous entry point for the v4 annotation pipeline.

    Loads config from defaults -> user YAML -> CLI dotlist overrides, then
    runs the pipeline.  All runtime inputs (image_dir, job_id, log_level)
    must be specified in the config's ``runtime:`` section.

    Parameters
    ----------
    user_config:
        Path to a user YAML config file that overrides the packaged defaults.
    overrides:
        List of OmegaConf dotlist overrides (e.g. ``["runtime.log_level=DEBUG"]``).
    """
    config = load_config(user_config=user_config, overrides=overrides)
    configure_logging(config.runtime.log_level, log_file=config.runtime.log_file)
    pipeline = AutoAnnotationPipelineV4(config)
    asyncio.run(pipeline.run())
