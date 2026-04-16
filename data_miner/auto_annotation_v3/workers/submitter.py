"""JobSubmitter — utility for seeding the pipeline's detect stream.

Usage::

    async with RedisMessageBroker.from_config(config) as broker:
        submitter = JobSubmitter(config, broker)
        n = await submitter.submit_directory("/data/images", job_id="run_001")
        print(f"Submitted {n} images")
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..checkpoint import CheckpointManager
from ..config import AutoAnnotationV3Config
from ..contracts import StageMessage
from .messaging import RedisMessageBroker

logger = logging.getLogger("data_miner.auto_annotation_v3.submitter")

# Image extensions considered by submit_directory.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


class JobSubmitter:
    """Submits images to the pipeline detect stream.

    Parameters
    ----------
    config:
        Full pipeline configuration (used for any future per-class routing).
    broker:
        A connected :class:`~.messaging.RedisMessageBroker` instance.
    checkpoint:
        Optional :class:`~..checkpoint.CheckpointManager`. When provided,
        images that already have all stages complete are skipped on
        submission (resume-safe).
    """

    def __init__(
        self,
        config: AutoAnnotationV3Config,
        broker: RedisMessageBroker,
        checkpoint: CheckpointManager | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.checkpoint = checkpoint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_images(
        self,
        image_paths: list[str],
        job_id: str,
    ) -> tuple[int, int]:
        """Submit image paths to the detect stream, skipping already-completed.

        Each image path becomes one :class:`~..contracts.StageMessage` with
        ``stage="detect"`` and ``attempt=0``. Images whose checkpoints show
        all stages complete are skipped (resume-safe re-runs).

        Parameters
        ----------
        image_paths:
            Absolute (or resolvable) paths to image files.
        job_id:
            Logical batch / run identifier shared by all submitted images.

        Returns
        -------
        tuple[int, int]
            ``(submitted, total_input)`` — number of images actually
            submitted and the total input count (including skipped).
            The caller should use *total_input* for the monitor so that
            same-job-id reruns wait for all images, not just new ones.
        """
        submitted = 0
        skipped = 0
        for path in image_paths:
            image_id = Path(path).stem
            if self.checkpoint and self.checkpoint.all_stages_complete(image_id):
                skipped += 1
                continue
            msg = StageMessage(
                image_id=image_id,
                image_path=str(path),
                job_id=job_id,
                stage="detect",
                attempt=0,
            )
            await self.broker.submit("detect", msg.model_dump())
            submitted += 1

        if skipped:
            logger.info(
                "Skipped %d already-completed image(s) for job '%s'",
                skipped, job_id,
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
        """Discover and submit all images from *image_dir*.

        Parameters
        ----------
        image_dir:
            Root directory to scan.
        job_id:
            Logical batch / run identifier.
        extensions:
            Set of lowercase file extensions to include.  Defaults to
            ``{".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}``.
        recursive:
            If ``True``, scan sub-directories as well (uses ``rglob``).

        Returns
        -------
        tuple[int, int]
            ``(submitted, total_input)`` — see :meth:`submit_images`.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        exts: frozenset[str] = (
            frozenset(e.lower() for e in extensions)
            if extensions is not None
            else _IMAGE_EXTENSIONS
        )

        glob_fn = image_dir.rglob if recursive else image_dir.glob
        paths = sorted(
            p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in exts
        )

        submitted, total_input = await self.submit_images([str(p) for p in paths], job_id)
        logger.info(
            "submit_directory: found %d image(s) in '%s'", len(paths), image_dir
        )
        return submitted, total_input
