"""Pipeline orchestrator: launches workers and manages the pipeline lifecycle."""

import asyncio
import logging
import signal
import uuid
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
            or f"job_{uuid.uuid4().hex[:8]}"
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
        self.submitter = JobSubmitter(config, self.broker)
        self.monitor = PipelineMonitor(self.broker, self.checkpoint)

        self._workers: list = []
        self._tasks: list = []

    def _create_workers(self) -> list:
        """Create worker instances based on config."""
        workers = []

        # Detect workers
        for i in range(self.config.workers.detect_count):
            w = DetectWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=self.output_writer,
                worker_id=f"detect-{i}",
            )
            workers.append(w)

        # Evaluate workers
        for i in range(self.config.workers.evaluate_count):
            w = EvaluateWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=self.output_writer,
                worker_id=f"evaluate-{i}",
            )
            workers.append(w)

        # Refine workers
        for i in range(self.config.workers.refine_count):
            w = RefineWorker(
                config=self.config,
                broker=self.broker,
                checkpoint_mgr=self.checkpoint,
                output_writer=self.output_writer,
                worker_id=f"refine-{i}",
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
        # Connect to Redis
        await self.broker.connect()

        try:
            # Resolve inputs: arg → config.runtime fallback
            image_dir = image_dir or self.config.runtime.image_dir
            image_paths = image_paths or list(self.config.runtime.image_paths)

            if image_paths:
                await self.submitter.submit_images(image_paths, self.job_id)
                total = len(image_paths)
            elif image_dir:
                total = await self.submitter.submit_directory(image_dir, self.job_id)
            else:
                raise ValueError(
                    "No input provided — set runtime.image_dir or "
                    "runtime.image_paths in config (or pass image_dir/"
                    "image_paths to run())"
                )

            logger.info("Submitted %d images for job %s", total, self.job_id)

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

            # Wait for completion
            completed = await self.monitor.wait_for_completion(
                total, poll_interval=5.0, timeout=None, print_progress=True
            )

            if completed:
                logger.info("Pipeline complete for job %s", self.job_id)
            else:
                logger.warning("Pipeline timed out for job %s", self.job_id)

            # Write summary
            status = await self.monitor.get_status()
            self.output_writer.write_summary(
                {
                    "job_id": self.job_id,
                    "total_images": total,
                    "status": status,
                    "completed": completed,
                }
            )

        finally:
            self._shutdown()
            await self.broker.close()

    def _shutdown(self) -> None:
        """Gracefully stop all workers."""
        logger.info("Shutting down workers...")
        for w in self._workers:
            w.stop()
        for t in self._tasks:
            t.cancel()

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
