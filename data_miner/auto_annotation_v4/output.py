"""Final pipeline output writers for auto_annotation_v4.

Writes YOLO label files, per-image audit traces, review-queue items,
a ``classes.txt`` manifest, and an aggregate ``summary.json``.

All file operations use atomic writes (write to ``.tmp``, then rename)
so a crashed worker never leaves partially-written output.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from . import utils

logger = logging.getLogger("data_miner.auto_annotation_v4.output")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via a sibling .tmp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialise *data* as JSON and write atomically to *path*."""
    _atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=False))


# ---------------------------------------------------------------------------
# OutputWriter
# ---------------------------------------------------------------------------


class OutputWriter:
    """Writes final pipeline outputs: YOLO labels, traces, and review queue.

    Directory layout under *job_dir*::

        labels/          YOLO .txt annotation files
        traces/          per-image audit trail JSON files
        review/          items queued for human review
        classes.txt      class id → class name mapping
        summary.json     aggregate job statistics

    Parameters
    ----------
    job_dir:
        Root output directory for the job.  Created on construction if absent.
    """

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = Path(job_dir)
        self.labels_dir = self.job_dir / "labels"
        self.traces_dir = self.job_dir / "traces"
        self.review_dir = self.job_dir / "review"

        for d in (self.labels_dir, self.traces_dir, self.review_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # YOLO labels
    # ------------------------------------------------------------------

    def write_yolo_labels(
        self,
        image_id: str,
        annotations: list,
        class_map: dict[str, int],
    ) -> Path:
        """Write accepted annotations as a YOLO-format label file.

        Parameters
        ----------
        image_id:
            Used as the stem of the output file (``labels/{image_id}.txt``).
        annotations:
            Objects with ``.class_name`` and ``.bbox`` attributes
            (e.g. :class:`~configs.contracts.FinalAnnotation`).
        class_map:
            Mapping ``class_name → class_id``.

        Returns
        -------
        Path
            Location of the written label file.
        """
        lines: list[str] = []
        for ann in annotations:
            class_id = class_map.get(ann.class_name)
            if class_id is None:
                logger.warning(
                    "Unknown class '%s' for image %s; skipping annotation.",
                    ann.class_name,
                    image_id,
                )
                continue
            lines.append(utils.annotation_to_yolo_line(class_id, ann.bbox))

        content = "\n".join(lines) + ("\n" if lines else "")
        path = self.labels_dir / f"{image_id}.txt"
        _atomic_write_text(path, content)
        logger.debug(
            "Wrote %d YOLO annotations for %s → %s", len(lines), image_id, path
        )
        return path

    # ------------------------------------------------------------------
    # Audit traces
    # ------------------------------------------------------------------

    def write_trace(self, image_id: str, trace_data: dict) -> Path:
        """Write the full per-image audit trail as JSON.

        Parameters
        ----------
        image_id:
            Used as the stem of the output file (``traces/{image_id}.json``).
        trace_data:
            Arbitrary dict capturing intermediate pipeline decisions (stage
            outputs, timing, verdicts, etc.).

        Returns
        -------
        Path
            Location of the written trace file.
        """
        path = self.traces_dir / f"{image_id}.json"
        _atomic_write_json(path, trace_data)
        logger.debug("Wrote trace for %s → %s", image_id, path)
        return path

    # ------------------------------------------------------------------
    # Review queue
    # ------------------------------------------------------------------

    def write_review(self, image_id: str, review_items: list) -> Path | None:
        """Write items requiring human review.

        Skips file creation when *review_items* is empty.

        Parameters
        ----------
        image_id:
            Used as the stem of the output file (``review/{image_id}.json``).
        review_items:
            List of dicts (or Pydantic models) describing candidates that need
            manual inspection.

        Returns
        -------
        Path or None
            Location of the written file, or ``None`` if there was nothing to write.
        """
        if not review_items:
            logger.debug("No review items for %s; skipping review file.", image_id)
            return None

        # Serialise Pydantic models if necessary
        serialisable = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in review_items
        ]

        path = self.review_dir / f"{image_id}.json"
        _atomic_write_json(path, serialisable)
        logger.debug(
            "Wrote %d review item(s) for %s → %s", len(review_items), image_id, path
        )
        return path

    # ------------------------------------------------------------------
    # Job-level files
    # ------------------------------------------------------------------

    def write_classes_file(self, class_map: dict[str, int]) -> Path:
        """Write ``classes.txt`` to *job_dir*, sorted by class id.

        Parameters
        ----------
        class_map:
            Mapping ``class_name → class_id``.

        Returns
        -------
        Path
            Location of the written file.
        """
        sorted_names = sorted(class_map, key=lambda n: class_map[n])
        content = "\n".join(sorted_names) + "\n"
        path = self.job_dir / "classes.txt"
        _atomic_write_text(path, content)
        logger.info("Wrote classes.txt (%d classes) → %s", len(sorted_names), path)
        return path

    def write_summary(self, stats: dict) -> Path:
        """Write aggregate job statistics to ``summary.json``.

        Parameters
        ----------
        stats:
            Dict of aggregate counters and metrics (e.g. total images processed,
            annotation counts per class, timing summaries).

        Returns
        -------
        Path
            Location of the written file.
        """
        path = self.job_dir / "summary.json"
        _atomic_write_json(path, stats)
        logger.info("Wrote summary.json → %s", path)
        return path
