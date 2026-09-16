"""Shared side-effect helpers for human-review writeback."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult

logger = logging.getLogger(__name__)


def append_human_review_trace(
    traces_dir: Path,
    image_id: str,
    result: HumanReviewResult,
) -> None:
    """Append a ``human_review`` block to ``traces/{image_id}.json``."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{image_id}.json"
    block = {
        "stage": "human_review",
        "ts": time.time(),
        "data": json.loads(result.model_dump_json()),
    }
    new_completion_id = result.ls_completion_id
    new_reviewed_at = getattr(result, "reviewed_at", None)

    existing: Any = None
    if trace_path.exists():
        try:
            existing = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None

    if not new_completion_id and not new_reviewed_at:
        logger.warning(
            "refusing trace append for %s: both ls_completion_id and reviewed_at are missing",
            image_id,
        )
        return

    if isinstance(existing, list):
        if _trace_already_recorded(existing, image_id, new_completion_id, new_reviewed_at):
            logger.info(
                "skip trace append for %s: completion %s already recorded",
                image_id,
                new_completion_id,
            )
            return
        existing.append(block)
        payload = existing
    elif isinstance(existing, dict):
        history = existing.setdefault("history", [])
        if isinstance(history, list) and _trace_already_recorded(
            history,
            image_id,
            new_completion_id,
            new_reviewed_at,
        ):
            logger.info(
                "skip trace append for %s: completion %s already recorded",
                image_id,
                new_completion_id,
            )
            return
        if isinstance(history, list):
            history.append(block)
        payload = existing
    else:
        payload = [block]

    tmp = trace_path.with_suffix(trace_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, trace_path)


def rewrite_yolo_label(
    labels_dir: Path,
    image_id: str,
    result: HumanReviewResult,
    classes_file: Path | None,
    *,
    dry_run: bool = False,
) -> None:
    """Rewrite ``labels/{image_id}.txt`` from human-review corrections."""
    class_to_id = _read_class_ids(classes_file)
    lines: list[str] = []
    for correction in result.corrections:
        if class_to_id and correction.class_name not in class_to_id:
            raise ValueError(
                f"class {correction.class_name!r} not in {classes_file}; "
                "fix the registry or the reviewer's label before retrying"
            )
        class_id = class_to_id.get(correction.class_name, 0)
        center_x = (correction.bbox.x1 + correction.bbox.x2) / 2.0
        center_y = (correction.bbox.y1 + correction.bbox.y2) / 2.0
        width = correction.bbox.x2 - correction.bbox.x1
        height = correction.bbox.y2 - correction.bbox.y1
        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    target = labels_dir / f"{image_id}.txt"
    payload = "\n".join(lines) + ("\n" if lines else "")
    if dry_run:
        logger.info("[dry-run] would write %d line(s) to %s", len(lines), target)
        return

    labels_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)


def _trace_already_recorded(
    history: list[Any],
    image_id: str,
    completion_id: int,
    reviewed_at: float | None,
) -> bool:
    """Return whether a trace history already contains this review result."""
    for entry in history:
        if not isinstance(entry, dict):
            continue
        data = entry.get("data") or {}
        if completion_id:
            if data.get("ls_completion_id") == completion_id:
                return True
        elif data.get("image_id") == image_id and data.get("reviewed_at") == reviewed_at:
            return True
    return False


def _read_class_ids(classes_file: Path | None) -> dict[str, int]:
    """Read a YOLO classes file into a class-name to class-id map."""
    class_to_id: dict[str, int] = {}
    if classes_file and classes_file.exists():
        text = classes_file.read_text(encoding="utf-8")
        if text.startswith("﻿"):
            text = text.lstrip("﻿")
        for index, line in enumerate(text.splitlines()):
            name = line.strip()
            if name:
                class_to_id[name] = index
        if not class_to_id:
            raise ValueError(
                f"classes file is empty or malformed: {classes_file} — "
                "without entries every box would silently collapse to class 0"
            )
    return class_to_id