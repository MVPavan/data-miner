"""Manual review system: human-in-the-loop layer over auto_annotation_v4.

Reads from and writes to the aa_v4 ``pipeline.db`` so reviewer corrections
land alongside the auto-pipeline's stage-level audit trail. See
``docs/review_system.md`` for the overall design.
"""
