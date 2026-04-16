"""Pipeline orchestrator: launches workers and manages the pipeline lifecycle."""

import asyncio
import logging
import signal
import urllib.request
from datetime import datetime
from pathlib import Path

from .config import AutoAnnotationV3Config, load_config, compute_config_hash
from .checkpoint import CheckpointManager
from .output import OutputWriter
from .workers.messaging import RedisMessageBroker
from .workers.submitter import JobSubmitter
from .workers.monitor import PipelineMonitor
from .stages.detect import DetectWorker
from .stages.evaluate import EvaluateWorker
from .stages.refine import RefineWorker
from .stages.finalize import FinalizeWorker
from .utils import configure_logging, get_logger

logger = get_logger("pipeline")


class AutoAnnotationPipelineV3:
    """Manages the full v3 annotation pipeline."""

    def __init__(self, config: AutoAnnotationV3Config, job_id: str | None = None):
        self.config = config
        # Prefer explicit job_id, then config.runtime.job_id, then autogen.
        self.job_id = (
            job_id
            or config.runtime.job_id
            or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Setup output directory
        self.job_dir = Path(config.output.job_dir) / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # Save frozen config as JSON (config.yaml extension kept for readability)
        (self.job_dir / "config.yaml").write_text(
            config.model_dump_json(indent=2)
        )

        # Initialize components
        self.checkpoint = CheckpointManager(self.job_dir / "checkpoints")
        self.output_writer = OutputWriter(self.job_dir)
        self.broker = RedisMessageBroker.from_config(config)
        self.submitter = JobSubmitter(config, self.broker, checkpoint=self.checkpoint)
        self.monitor = PipelineMonitor(self.broker, self.checkpoint)

        self._workers: list = []
        self._tasks: list = []
        self._monitor_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Preflight health checks
    # ------------------------------------------------------------------

    def _check_endpoint(self, url: str, timeout: float = 3.0) -> bool:
        """Return True if *url* responds with HTTP 200."""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def preflight(self) -> None:
        """Verify all required servers are online before the pipeline starts.

        Checks:
        - Every enabled detector server (/health endpoint)
        - VLM server (/health endpoint)
        - Redis connectivity

        Raises RuntimeError with a summary of all failures.
        """
        logger.info("Running preflight health checks …")
        failures: list[str] = []

        # 1. Detector servers
        for name, cfg in self.config.servers.enabled_detectors().items():
            url = f"http://localhost:{cfg.port}/health"
            if self._check_endpoint(url):
                logger.info("  ✓ %s (port %d)", name.value, cfg.port)
            else:
                msg = f"Detector '{name.value}' not reachable at localhost:{cfg.port}/health"
                logger.error("  ✗ %s", msg)
                failures.append(msg)

        # 2. VLM server
        vlm_base = self.config.servers.vlm.url.rstrip("/")
        # vLLM exposes /health at the base URL (not under /v1)
        vlm_health = vlm_base.replace("/v1", "") + "/health"
        if self._check_endpoint(vlm_health):
            logger.info("  ✓ VLM (%s)", vlm_health)
        else:
            msg = f"VLM not reachable at {vlm_health}"
            logger.error("  ✗ %s", msg)
            failures.append(msg)

        # 3. Redis
        try:
            import redis
            r = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                socket_connect_timeout=3,
            )
            r.ping()
            r.close()
            logger.info(
                "  ✓ Redis (%s:%d)", self.config.redis.host, self.config.redis.port
            )
        except Exception as exc:
            msg = f"Redis not reachable at {self.config.redis.host}:{self.config.redis.port} — {exc}"
            logger.error("  ✗ %s", msg)
            failures.append(msg)

        if failures:
            summary = "\n  - ".join(failures)
            raise RuntimeError(
                f"Preflight failed — {len(failures)} server(s) unreachable:\n  - {summary}\n"
                "Start the missing servers and retry."
            )
        logger.info("Preflight passed — all servers healthy.")

    def _create_workers(self) -> list:
        """Create worker instances based on config."""
        workers = []

        # Detect workers (output_writer=None — finalize is the single sink)
        for i in range(self.config.workers.detect_count):
            w = DetectWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=None,
                worker_id=f"detect-{i}",
                job_id=self.job_id,
            )
            workers.append(w)

        # Evaluate workers (output_writer=None — finalize is the single sink)
        for i in range(self.config.workers.evaluate_count):
            w = EvaluateWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=None,
                worker_id=f"evaluate-{i}",
                job_id=self.job_id,
            )
            workers.append(w)

        # Refine workers
        for i in range(self.config.workers.refine_count):
            w = RefineWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=None,   # outputs are written by finalize
                worker_id=f"refine-{i}",
                job_id=self.job_id,
            )
            workers.append(w)

        # Finalize workers — sole owner of YOLO label / trace / review writes.
        for i in range(self.config.workers.finalize_count):
            w = FinalizeWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=self.output_writer,
                worker_id=f"finalize-{i}",
                job_id=self.job_id,
            )
            workers.append(w)

        return workers

    async def run(
        self,
        image_paths: list[str] | None = None,
        image_dir: str | None = None,
    ) -> None:
        """Run the full pipeline.

        Input sources are resolved in order of priority:
        1. ``image_paths`` argument
        2. ``image_dir`` argument
        3. ``config.runtime.image_paths``
        4. ``config.runtime.image_dir``
        """
        # Preflight: verify all servers are reachable before committing.
        self.preflight()

        # Connect to Redis
        await self.broker.connect()

        try:
            # Resolve inputs: arg → config.runtime fallback
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
                logger.info("All images already complete for job %s — nothing to do", self.job_id)
                return

            logger.info("Submitted %d/%d images for job %s", submitted, total_input, self.job_id)

            # Write classes file
            class_map = {c.name: c.id for c in self.config.classes}
            self.output_writer.write_classes_file(class_map)

            # Launch workers
            self._workers = self._create_workers()
            self._tasks = [
                asyncio.create_task(w.run()) for w in self._workers
            ]

            # Setup signal handling for graceful shutdown
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._shutdown)

            # Wait for completion — use total_input (not submitted) so
            # same-job-id reruns wait for ALL images, not just new ones.
            # Wrapped in a task so _shutdown() can cancel it on Ctrl+C.
            self._monitor_task = asyncio.create_task(
                self.monitor.wait_for_completion(
                    total_input, poll_interval=5.0, timeout=None, print_progress=True
                )
            )
            try:
                completed = await self._monitor_task
            except asyncio.CancelledError:
                logger.info("Monitor cancelled — shutting down")
                completed = False

            if completed:
                logger.info("Pipeline complete for job %s", self.job_id)
            else:
                logger.warning("Pipeline did not complete for job %s", self.job_id)

            # Write summary
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
            self._shutdown()
            await self.broker.close()

    def _shutdown(self) -> None:
        """Gracefully stop all workers and the monitor."""
        logger.info("Shutting down workers...")
        for w in self._workers:
            w.stop()
        for t in self._tasks:
            t.cancel()
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    async def run_single_image(self, image_path: str) -> None:
        """Run pipeline for a single image (convenience method)."""
        await self.run(image_paths=[image_path])


def run_pipeline(
    user_config: str | None = None,
    overrides: list[str] | None = None,
) -> None:
    """Synchronous entry point for the v3 annotation pipeline.

    Loads config from defaults → user YAML → CLI dotlist overrides, then runs
    the pipeline. All runtime inputs (image_dir, job_id, log_level) must be
    specified in the config's ``runtime:`` section.
    """
    config = load_config(user_config=user_config, overrides=overrides)
    configure_logging(config.runtime.log_level, log_file=config.runtime.log_file)
    pipeline = AutoAnnotationPipelineV3(config)
    asyncio.run(pipeline.run())
