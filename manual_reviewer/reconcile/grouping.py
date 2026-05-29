"""Group survivor frames into reconciliation cohorts.

The reconciler only propagates static-object detections within a *group* —
the assumption being that frames in the same group share roughly the same
camera pose / scene background.

Default strategy: ``clip_id`` extracted from the filename via a regex that
strips the trailing frame index (``foo_clip_0001.jpg``, ``foo_clip_0002.jpg``
→ both belong to group ``foo_clip``). Frames whose filename doesn't match the
regex fall back to a singleton group keyed by image_id.

A user can override the regex via CLI, or pass ``strategy="all"`` to throw
every survivor into one group (suitable for static-camera batches whose
filenames don't carry a clip prefix).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Iterable

__all__ = ["group_images", "DEFAULT_CLIP_REGEX"]


# Matches "anything_<digits>.<ext>" — captures the prefix as group(1).
# Examples that match:
#   warehouse_cam2_0001.jpg → "warehouse_cam2"
#   loading_dock_42_frame_007.png → "loading_dock_42_frame"
# Examples that don't match (single-image, no frame index):
#   poster.jpg, scan_2024_01_15.png (no _<digits>.ext suffix)
DEFAULT_CLIP_REGEX = re.compile(r"^(.+?)[_\-]\d+\.[A-Za-z0-9]+$")


def group_images(
    images: Iterable[tuple[str, str]],
    *,
    strategy: str = "clip_id",
    clip_regex: re.Pattern[str] | None = None,
) -> dict[str, list[str]]:
    """Group ``(image_id, image_path)`` tuples by reconciliation cohort.

    Returns ``{group_id: [image_id, ...]}`` with deterministic ordering
    (sorted by image_id within each group, groups sorted by id).

    ``strategy``:
      - ``"clip_id"`` (default): regex match on filename basename, falls back
        to image_id for unmatched names.
      - ``"all"``: one big group keyed ``"all"``.
      - ``"per_image"``: each image gets its own group (disables propagation,
        useful as a no-op control).
    """
    if strategy not in {"clip_id", "all", "per_image"}:
        raise ValueError(f"unknown grouping strategy: {strategy}")

    rx = clip_regex or DEFAULT_CLIP_REGEX
    out: dict[str, list[str]] = defaultdict(list)

    for image_id, image_path in images:
        if strategy == "all":
            key = "all"
        elif strategy == "per_image":
            key = image_id
        else:  # clip_id
            basename = PurePosixPath(image_path).name if image_path else image_id
            m = rx.match(basename)
            key = m.group(1) if m else image_id

        out[key].append(image_id)

    return {gid: sorted(ids) for gid, ids in sorted(out.items())}
