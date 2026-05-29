"""Filename-based clip grouping.

aa_v4 image_ids follow the convention ``<clip_name>_f<frame_index>`` (e.g.
``Caifu_Center_Fewer_2_f00516``). Two image_ids belong to the same clip iff
their prefixes — everything before the trailing ``_f<digits>`` — match.

Used by:
  * ``scripts/build_tasks.py`` for per-clip diversity sampling.
  * ``ml_backend/smart_track_lib.py`` for sibling-frame discovery during
    cross-frame tracker propagation.
"""

from __future__ import annotations

import re

_CLIP_SUFFIX = re.compile(r"_f\d+$")


def clip_prefix(image_id: str) -> str:
    """Image_id with the trailing ``_f<digits>`` suffix stripped.

    Falls back to the full image_id when the suffix doesn't match (so
    images that don't follow the convention each form their own
    one-element clip rather than collapsing into a generic bucket).
    """
    return _CLIP_SUFFIX.sub("", image_id) or image_id
