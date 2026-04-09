"""Finalize stage: compile final annotations, YOLO labels, and full trace."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import AutoAnnotationV2Config, OutputConfig
from ..contracts import (
    Candidate,
    CandidateStatus,
    DetailedVerdict,
    FinalAction,
    FinalAnnotation,
    ImageTrace,
    PipelineResult,
    RefinementAction,
    ScreeningVerdict,
    VLMDecision,
)
from ..log_utils import get_logger
from ..utils import annotation_to_yolo_line

logger = get_logger(__name__)


def _resolve_final_action(
    candidate: Candidate,
    screening: list[ScreeningVerdict],
    detailed: list[DetailedVerdict],
    validation_screening: list[ScreeningVerdict],
    validation_detailed: list[DetailedVerdict],
) -> tuple[FinalAction, float, list[str]]:
    """Determine the final action for a candidate based on all VLM verdicts."""
    cid = candidate.candidate_id
    trace: list[str] = [f"proposed by {candidate.source_model}"]

    # Check validation verdicts first (for refined candidates)
    for v in validation_detailed:
        if v.candidate_id == cid:
            trace.append(f"validation_detailed: {v.decision.value} ({v.reasoning})")
            if v.decision == VLMDecision.ACCEPT:
                return FinalAction.ACCEPT, v.confidence, trace
            elif v.decision == VLMDecision.REJECT:
                return FinalAction.REJECT, v.confidence, trace
            else:
                return FinalAction.HUMAN_REVIEW, v.confidence, trace

    for v in validation_screening:
        if v.candidate_id == cid:
            trace.append(f"validation_screening: {v.decision.value} ({v.reasoning})")
            if v.decision == VLMDecision.ACCEPT:
                return FinalAction.ACCEPT, v.confidence, trace
            elif v.decision == VLMDecision.REJECT:
                return FinalAction.REJECT, v.confidence, trace

    # Check original detailed verdicts
    for v in detailed:
        if v.candidate_id == cid:
            trace.append(f"detailed: {v.decision.value} ({v.reasoning})")
            if v.decision == VLMDecision.ACCEPT:
                return FinalAction.ACCEPT, v.confidence, trace
            elif v.decision == VLMDecision.REJECT:
                return FinalAction.REJECT, v.confidence, trace
            else:
                return FinalAction.HUMAN_REVIEW, v.confidence, trace

    # Check screening verdicts
    for v in screening:
        if v.candidate_id == cid:
            trace.append(f"screening: {v.decision.value} ({v.reasoning})")
            if v.decision == VLMDecision.ACCEPT:
                return FinalAction.ACCEPT, v.confidence, trace
            elif v.decision == VLMDecision.REJECT:
                return FinalAction.REJECT, v.confidence, trace
            else:
                return FinalAction.HUMAN_REVIEW, v.confidence, trace

    # No VLM verdict at all — escalate to human review
    trace.append("no VLM verdict found, escalating")
    return FinalAction.HUMAN_REVIEW, 0.0, trace


def build_final_annotations(
    candidates: list[Candidate],
    screening: list[ScreeningVerdict],
    detailed: list[DetailedVerdict],
    validation_screening: list[ScreeningVerdict],
    validation_detailed: list[DetailedVerdict],
) -> list[FinalAnnotation]:
    """Build final annotations for all candidates."""
    annotations: list[FinalAnnotation] = []
    for cand in candidates:
        action, confidence, trace = _resolve_final_action(
            cand, screening, detailed, validation_screening, validation_detailed
        )

        # Check relabel from detailed verdict
        relabel = None
        for v in [*validation_detailed, *detailed]:
            if v.candidate_id == cand.candidate_id and v.relabel_to:
                relabel = v.relabel_to
                break

        annotations.append(FinalAnnotation(
            candidate_id=cand.candidate_id,
            class_name=relabel or cand.class_name,
            bbox=cand.bbox,
            action=action,
            confidence=confidence,
            source_model=cand.source_model,
            reasoning_trace=trace,
            was_refined=cand.status == CandidateStatus.REFINED,
            original_bbox=None,  # Could track original if needed
        ))

    return annotations


def build_yolo_lines(
    annotations: list[FinalAnnotation],
    class_names: list[str],
) -> list[str]:
    """Generate YOLO format lines for accepted annotations."""
    label_map = {name: idx for idx, name in enumerate(class_names)}
    lines: list[str] = []
    for ann in annotations:
        if ann.action != FinalAction.ACCEPT:
            continue
        class_idx = label_map.get(ann.class_name)
        if class_idx is None:
            logger.warning(
                "Accepted annotation %s has unmapped class '%s'",
                ann.candidate_id, ann.class_name,
            )
            continue
        lines.append(annotation_to_yolo_line(ann, class_idx))
    return lines


def build_pipeline_result(
    image_path: str,
    trace: ImageTrace,
    partial: bool = False,
) -> PipelineResult:
    """Build the final pipeline result from the trace."""
    accepted = [a for a in trace.final_annotations if a.action == FinalAction.ACCEPT]
    rejected = [a for a in trace.final_annotations if a.action == FinalAction.REJECT]
    human_review = [a for a in trace.final_annotations if a.action == FinalAction.HUMAN_REVIEW]

    class_names = list({a.class_name for a in trace.final_annotations})
    class_names.sort()
    yolo_lines = build_yolo_lines(trace.final_annotations, class_names)

    return PipelineResult(
        image_path=image_path,
        accepted=accepted,
        rejected=rejected,
        human_review=human_review,
        yolo_lines=yolo_lines,
        trace=trace,
        partial=partial,
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def save_result(
    result: PipelineResult,
    class_names: list[str],
    output_dir: Path,
    output_cfg: OutputConfig,
) -> None:
    """Write YOLO labels, trace, and review queue to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem

    # YOLO labels
    if output_cfg.save_labels:
        label_dir = output_dir / output_cfg.label_dirname
        label_dir.mkdir(parents=True, exist_ok=True)
        yolo_lines = build_yolo_lines(result.accepted, class_names)
        (label_dir / f"{stem}.txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )

    # Full trace
    if output_cfg.save_traces:
        trace_dir = output_dir / output_cfg.trace_dirname
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{stem}.json").write_text(
            json.dumps(result.trace.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Review queue
    review_items = result.human_review
    if output_cfg.save_review_queue and (review_items or result.partial or result.trace.failures):
        review_dir = output_dir / output_cfg.review_dirname
        review_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_path": result.image_path,
            "candidate_ids": [a.candidate_id for a in review_items],
            "partial": result.partial,
            "failure_count": len(result.trace.failures),
        }
        (review_dir / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
