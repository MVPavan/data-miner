"""Single source of truth for every enumerated value in auto_annotation_v4.

No free-form strings anywhere in the pipeline — every status, stage, detector
name, verdict, and disposition is defined here and imported by contracts,
workers, and DB helpers alike.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

__all__ = [
    "Stage",
    "STAGE_ORDER",
    "FilterContext",
    "WorkStatus",
    "ImageStatus",
    "DetectorName",
    "CandidateStatus",
    "FinalAction",
    "RefineAction",
    "RefineOutcome",
    "Verdict",
    "BboxSource",
    "DropReason",
    "TiebreakField",
    "ClassTier",
]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


class Stage(StrEnum):
    """Pipeline execution stages stored in checkpoint metadata and Redis messages."""

    DETECT = "detect"
    FILTER = "filter"
    EVALUATE = "evaluate"
    REFINE = "refine"
    FINALIZE = "finalize"
    # Event-driven, written by manual_reviewer/scripts/export_to_aa_v4.py.
    # Deliberately NOT in STAGE_ORDER — the auto pipeline must complete without it.
    HUMAN_REVIEW = "human_review"
    DONE = "done"


STAGE_ORDER: list[Stage] = [
    Stage.DETECT,
    Stage.FILTER,
    Stage.EVALUATE,
    Stage.REFINE,
    Stage.FINALIZE,
]


class FilterContext(StrEnum):
    """Which pipeline point a filter was invoked from."""

    POST_DETECT = "post_detect"
    POST_REVIEW = "post_review"
    POST_REFINE = "post_refine"
    PRE_FINALIZE = "pre_finalize"


# ---------------------------------------------------------------------------
# Queue / image tracking
# ---------------------------------------------------------------------------


class WorkStatus(StrEnum):
    """Status values for rows in the work_queue table."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"


class ImageStatus(StrEnum):
    """Per-image lifecycle status stored in image_meta."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class DetectorName(StrEnum):
    """Canonical detector identifiers matching YAML keys in servers.detectors."""

    GROUNDING_DINO = "grounding_dino"
    FALCON = "falcon"
    SAM3 = "sam3"
    SAM3_DART = "sam3_dart"
    OWLVIT2 = "owlvit2"
    OMDET_TURBO = "omdet_turbo"

    @property
    def is_sam3_family(self) -> bool:
        """True for SAM3 and SAM3_DART which share the same wire contract."""
        return self in (DetectorName.SAM3, DetectorName.SAM3_DART)


# ---------------------------------------------------------------------------
# Candidate lifecycle
# ---------------------------------------------------------------------------


class CandidateStatus(StrEnum):
    """Tracks each candidate from initial proposal through final disposition."""

    PROPOSED = "proposed"
    FILTERED_OUT = "filtered_out"
    AUTO_ACCEPTED = "auto_accepted"
    NEEDS_EVALUATION = "needs_evaluation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REFINED = "refined"


# ---------------------------------------------------------------------------
# Final / refine actions and outcomes
# ---------------------------------------------------------------------------


class FinalAction(StrEnum):
    """Terminal disposition applied to each annotation in the finalize stage."""

    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"


class RefineAction(StrEnum):
    """Per-prompt VLM directive: skip the prompt or propose a region."""

    SKIP = "skip"
    PROPOSE = "propose"


class RefineOutcome(StrEnum):
    """Outcome of a single prompt step inside the refine inner loop."""

    SKIPPED = "skipped"
    SAM_NO_MASK = "sam_no_mask"
    PRESENCE_FAILED = "presence_failed"
    MERGE_FAILED = "merge_failed"
    MERGED = "merged"
    VLM_ERROR = "vlm_error"
    SAM_ERROR = "sam_error"


# ---------------------------------------------------------------------------
# Verdicts and bbox selection
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    """Tri-state verdict used by evaluate and refine adjudication."""

    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


class BboxSource(StrEnum):
    """Which bounding box to keep after refinement."""

    REFINED = "refined"
    ORIGINAL = "original"


# ---------------------------------------------------------------------------
# Finalize drop reasons
# ---------------------------------------------------------------------------


class DropReason(StrEnum):
    """Reason a candidate was dropped during filter / evaluate / finalize."""

    SOURCE_MODEL = "source_model"
    GEOMETRIC_FILTER = "geometric_filter"
    SCORE_FLOOR = "score_floor"
    DEDUP = "dedup"
    CROSS_CLASS = "cross_class"
    PER_CLASS_CAP = "per_class_cap"
    REJECTED_UPSTREAM = "rejected_upstream"
    HEAD_WITHOUT_PERSON = "head_without_person"
    CLASS_AGNOSTIC_NMS = "class_agnostic_nms"
    # ---- Evaluate-stage rejects ----
    VLM_LOW_CONFIDENCE = "vlm_low_confidence"
    """class_confidence below ``evaluate.reject_below``."""
    VLM_OTHER_CLASS = "vlm_other_class"
    """VLM returned "other"/"unknown"/"none" as detected_class."""
    VLM_UNKNOWN_CLASS = "vlm_unknown_class"
    """VLM named a class that doesn't resolve through the alias map."""
    VLM_BBOX_UNUSABLE = "vlm_bbox_unusable"
    """Class trusted (≥accept_above) but bbox_score below
    ``evaluate.bbox_reject_below`` — bbox on wrong region or object missing."""
    VLM_MALFORMED = "vlm_malformed"
    """Could not parse VLM JSON into a VLMVerdict (telemetry for prompt /
    model drift)."""


# ---------------------------------------------------------------------------
# Class tiers
# ---------------------------------------------------------------------------


class TiebreakField(StrEnum):
    """Fields used in the IoU dedup tiebreak cascade."""

    AGREEMENT = "agreement"
    MODEL_PRIORITY = "model_priority"
    SCORE = "score"


class ClassTier(IntEnum):
    """Priority tier for annotation classes, drives per-class caps and routing."""

    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
