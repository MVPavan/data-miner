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

    def __init__(self, config: AutoAnnotationV3Config, job_id: str = None):
        self.config = config
        self.job_id = job_id or f"job_{uuid.uuid4().hex[:8]}"

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
        image_paths: list[str] = None,
        image_dir: str = None,
    ) -> None:
        """Run the full pipeline."""
        # Connect to Redis
        await self.broker.connect()

        try:
            # Submit images
            if image_dir:
                total = await self.submitter.submit_directory(image_dir, self.job_id)
            elif image_paths:
                await self.submitter.submit_images(image_paths, self.job_id)
                total = len(image_paths)
            else:
                raise ValueError("Must provide image_paths or image_dir")

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
    config_path: str = None,
    image_dir: str = None,
    image_paths: list[str] = None,
    job_id: str = None,
    log_level: str = "INFO",
) -> None:
    """Synchronous entry point for the v3 annotation pipeline."""
    configure_logging(log_level)
    config = load_config(config_path)
    pipeline = AutoAnnotationPipelineV3(config, job_id)
    asyncio.run(pipeline.run(image_paths=image_paths, image_dir=image_dir))
