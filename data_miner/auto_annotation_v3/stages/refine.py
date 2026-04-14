"""Stage 3: Refine — SAM refinement with VLM-provided point coordinates."""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any

import aiohttp

from ..workers.base import StageWorker
from ..contracts import (
    StageMessage,
    DetectResult,
    EvaluateResult,
    RefineResult,
    RefinementResult,
    RefinementInstruction,
    RefinementNeeded,
    FinalAnnotation,
    FinalAction,
    BoundingBox,
    Candidate,
)
from ..config import AutoAnnotationV3Config
from ..checkpoint import CheckpointManager
from ..output import OutputWriter
from ..utils import bbox_iou, bbox_to_pixels, pixels_to_bbox

logger = logging.getLogger("data_miner.auto_annotation_v3.refine")


class RefineWorker(StageWorker):
    """Stage 3: SAM refinement driven by VLM point/box instructions from evaluate."""

    stage = "refine"

    def __init__(
        self,
        config: AutoAnnotationV3Config,
        broker: Any,
        checkpoint_mgr: CheckpointManager,
        output_writer: OutputWriter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, broker, checkpoint_mgr, **kwargs)
        self.output_writer = output_writer

    # ------------------------------------------------------------------
    # Core process method (called by StageWorker.run loop)
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> StageMessage | None:
        t0 = time.monotonic()

        # 1. Load prior-stage checkpoints — both are required to proceed.
        detect: DetectResult | None = self.load_checkpoint(
            msg.image_id, "detect", DetectResult
        )
        evaluate: EvaluateResult | None = self.load_checkpoint(
            msg.image_id, "evaluate", EvaluateResult
        )

        if detect is None or evaluate is None:
            raise RuntimeError(
                f"Missing checkpoints for {msg.image_id}: "
                f"detect={detect is not None}, evaluate={evaluate is not None}"
            )

        image_w, image_h = detect.image_size[0], detect.image_size[1]
        sam_port = self.config.servers.sam3.port

        # Build a fast lookup: candidate_id → Candidate
        cand_by_id: dict[str, Candidate] = {
            c.candidate_id: c for c in detect.candidates
        }

        # 2. Process each candidate that needs refinement.
        results: list[RefinementResult] = []
        sam_calls = 0
        instructions_used: list[RefinementInstruction] = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            for ref_needed in evaluate.refinement_needed:
                # EvaluateResult.refinement_needed is list[RefinementNeeded] —
                # Pydantic deserialises these as typed objects.
                cand_id: str = ref_needed.candidate_id

                cand = cand_by_id.get(cand_id)
                if cand is None:
                    self.logger.warning(
                        "Candidate %s not found in detect results — skipping", cand_id
                    )
                    continue

                # refinement_instructions is dict[str, RefinementInstruction]
                instr: RefinementInstruction | None = (
                    evaluate.refinement_instructions.get(cand_id)
                )
                if instr is None:
                    self.logger.warning(
                        "No refinement instruction for %s — keeping original bbox",
                        cand_id,
                    )
                    results.append(
                        RefinementResult(
                            candidate_id=cand_id,
                            original_bbox=cand.bbox,
                            refined_bbox=cand.bbox,
                            iou_with_original=1.0,
                            accepted=True,
                            method="fallback_no_instruction",
                        )
                    )
                    continue

                instructions_used.append(instr)

                try:
                    refined_bbox = await self._call_sam_refine(
                        session, sam_port, msg.image_path, cand, instr, image_w, image_h
                    )
                    sam_calls += 1

                    if refined_bbox is not None:
                        iou = bbox_iou(cand.bbox, refined_bbox)

                        # IoU-based auto-accept/reject.
                        if iou < self.config.refinement.reject_iou:
                            # SAM drifted too far from the original box — reject.
                            accepted = False
                            self.logger.warning(
                                "Refinement rejected for %s: IoU=%.3f < reject_iou=%.3f "
                                "(SAM drifted too far)",
                                cand_id, iou, self.config.refinement.reject_iou,
                            )
                        elif iou >= self.config.refinement.auto_accept_iou:
                            accepted = True
                        else:
                            # IoU is between reject and auto-accept thresholds.
                            # Default to accepting — downstream review can flag it.
                            accepted = True

                        results.append(
                            RefinementResult(
                                candidate_id=cand_id,
                                original_bbox=cand.bbox,
                                refined_bbox=refined_bbox,
                                iou_with_original=iou,
                                accepted=accepted,
                                method="sam_point",
                            )
                        )
                    else:
                        # SAM returned no mask/box — keep original.
                        self.logger.info(
                            "SAM returned no box for %s — keeping original bbox", cand_id
                        )
                        results.append(
                            RefinementResult(
                                candidate_id=cand_id,
                                original_bbox=cand.bbox,
                                refined_bbox=cand.bbox,
                                iou_with_original=1.0,
                                accepted=True,
                                method="fallback_original",
                            )
                        )

                except Exception as exc:
                    self.logger.error(
                        "SAM refine failed for %s: %s", cand_id, exc, exc_info=True
                    )
                    # Keep original bbox on any server error.
                    results.append(
                        RefinementResult(
                            candidate_id=cand_id,
                            original_bbox=cand.bbox,
                            refined_bbox=cand.bbox,
                            iou_with_original=1.0,
                            accepted=True,
                            method="fallback_error",
                        )
                    )

        stage_timing_ms = (time.monotonic() - t0) * 1000

        # 3. Build and save the refine checkpoint.
        refine_result = RefineResult(
            image_id=msg.image_id,
            refinement_instructions=instructions_used,
            results=results,
            vlm_calls=0,        # No VLM calls in refine — instructions came from evaluate.
            sam_calls=sam_calls,
            prompt_used=None,   # Refine stage uses no prompt template.
            stage_timing_ms=stage_timing_ms,
        )
        self.save_checkpoint(msg.image_id, "refine", refine_result)

        self.logger.info(
            "%s: refinement done — %d candidates, %d SAM calls, %.0f ms",
            msg.image_id, len(results), sam_calls, stage_timing_ms,
        )

        # 4. Write final outputs (YOLO labels, audit trace, review queue).
        if self.output_writer is not None:
            self._write_final_output(msg.image_id, detect, evaluate, refine_result)

        # 5. Forward to the done stream.
        return msg.forward("done")

    # ------------------------------------------------------------------
    # SAM3 HTTP call
    # ------------------------------------------------------------------

    async def _call_sam_refine(
        self,
        session: aiohttp.ClientSession,
        sam_port: int,
        image_path: str,
        cand: Candidate,
        instr: RefinementInstruction,
        image_w: int,
        image_h: int,
    ) -> BoundingBox | None:
        """POST to the SAM3 /predict endpoint in *refine* mode.

        Sends the original bounding box (as pixel coordinates) plus the
        VLM-provided foreground point.  Returns a normalised BoundingBox,
        or None when SAM yields no usable result.
        """
        # Convert normalised bbox → pixel coords expected by SAM3.
        px1, py1, px2, py2 = bbox_to_pixels(cand.bbox, image_w, image_h)

        url = f"http://localhost:{sam_port}/predict"
        payload: dict = {
            "image_path": str(image_path),
            "mode": "refine",
            "bbox": [px1, py1, px2, py2],
            "point_label": 1,   # 1 = foreground
        }

        # Include point coords only when the instruction carries them.
        if instr.point_x is not None and instr.point_y is not None:
            payload["point"] = [instr.point_x, instr.point_y]

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data: dict = await resp.json()

        # SAM3 returns {"box": [x1, y1, x2, y2]} in pixel coords.
        box = data.get("box")
        if box and len(box) == 4:
            norm = pixels_to_bbox(
                int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                image_w, image_h,
            )
            return BoundingBox(**norm)

        return None

    # ------------------------------------------------------------------
    # Final output writing
    # ------------------------------------------------------------------

    def _write_final_output(
        self,
        image_id: str,
        detect: DetectResult,
        evaluate: EvaluateResult,
        refine: RefineResult,
    ) -> None:
        """Write YOLO labels, per-image audit trace, and review queue items."""
        class_map: dict[str, int] = {c.name: c.id for c in self.config.classes}

        # Build a fast lookup: candidate_id → RefinementResult (accepted only).
        refined_bboxes: dict[str, BoundingBox] = {}
        refine_by_id: dict[str, RefinementResult] = {}
        for r in refine.results:
            refine_by_id[r.candidate_id] = r
            if r.accepted and r.refined_bbox is not None:
                refined_bboxes[r.candidate_id] = r.refined_bbox

        accepted_set: set[str] = set(evaluate.accepted)
        rejected_set: set[str] = set(evaluate.rejected)
        refine_set: set[str] = {r.candidate_id for r in refine.results}

        annotations: list[FinalAnnotation] = []
        review_items: list[dict] = []

        for cand in detect.candidates:
            # Apply any VLM-issued relabelling (e.g. person → pedestrian).
            cls_name: str = evaluate.relabels.get(cand.candidate_id, cand.class_name)

            if cand.candidate_id in accepted_set:
                # VLM-accepted candidates: use refined bbox if one exists,
                # otherwise keep the original.
                bbox = refined_bboxes.get(cand.candidate_id, cand.bbox)
                was_refined = cand.candidate_id in refined_bboxes

                annotations.append(
                    FinalAnnotation(
                        candidate_id=cand.candidate_id,
                        class_name=cls_name,
                        class_id=class_map.get(cls_name, -1),
                        bbox=bbox,
                        confidence=cand.score,
                        action=FinalAction.ACCEPT,
                        source_model=cand.source_model,
                        was_refined=was_refined,
                        trace=self._build_trace(cand, evaluate, refine),
                    )
                )

            elif cand.candidate_id in rejected_set:
                # VLM-rejected: drop entirely — do not emit a label.
                continue

            elif cand.candidate_id in refine_set:
                # Went through SAM refinement (was in evaluate.refinement_needed).
                ref_result = refine_by_id[cand.candidate_id]

                if ref_result.accepted:
                    bbox = (
                        ref_result.refined_bbox
                        if ref_result.refined_bbox is not None
                        else cand.bbox
                    )
                    annotations.append(
                        FinalAnnotation(
                            candidate_id=cand.candidate_id,
                            class_name=cls_name,
                            class_id=class_map.get(cls_name, -1),
                            bbox=bbox,
                            confidence=cand.score,
                            action=FinalAction.ACCEPT,
                            source_model=cand.source_model,
                            was_refined=True,
                            trace=self._build_trace(cand, evaluate, refine),
                        )
                    )
                else:
                    # SAM drifted too far — send to human review.
                    review_items.append(
                        {
                            "candidate_id": cand.candidate_id,
                            "class_name": cls_name,
                            "bbox": cand.bbox.model_dump(),
                            "reason": "refinement_rejected",
                            "iou_with_original": ref_result.iou_with_original,
                            "original_bbox": ref_result.original_bbox.model_dump(),
                            "refined_bbox": (
                                ref_result.refined_bbox.model_dump()
                                if ref_result.refined_bbox is not None
                                else None
                            ),
                            "method": ref_result.method,
                        }
                    )
            # Candidates that are neither accepted, rejected, nor in refine_set
            # were not seen by the evaluate stage (e.g. auto-accepted in detect
            # and never VLM-evaluated).  They are silently skipped here — the
            # detect stage's own output writer handles auto-accepted candidates.

        self.output_writer.write_yolo_labels(image_id, annotations, class_map)

        self.output_writer.write_trace(
            image_id,
            {
                "image_id": image_id,
                "stages": ["detect", "evaluate", "refine"],
                "detect": detect.model_dump(mode="json"),
                "evaluate": evaluate.model_dump(mode="json"),
                "refine": refine.model_dump(mode="json"),
                "annotations": [a.model_dump(mode="json") for a in annotations],
            },
        )

        if review_items:
            self.output_writer.write_review(image_id, review_items)

        self.logger.info(
            "%s: wrote %d annotations, %d review items",
            image_id, len(annotations), len(review_items),
        )

    # ------------------------------------------------------------------
    # Trace builder
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        cand: Candidate,
        evaluate: EvaluateResult,
        refine: RefineResult,
    ) -> list[str]:
        """Build a human-readable audit trail for one candidate."""
        trace: list[str] = []

        # VLM verdict (confidence, class, bbox quality).
        verdict = next(
            (v for v in evaluate.verdicts if v.candidate_id == cand.candidate_id),
            None,
        )
        if verdict is not None:
            trace.append(
                f"vlm: class={verdict.correct_class} "
                f"conf={verdict.confidence:.2f} "
                f"quality={verdict.bbox_quality.value}"
            )

        # Relabelling.
        new_label = evaluate.relabels.get(cand.candidate_id)
        if new_label is not None and new_label != cand.class_name:
            trace.append(f"relabel: {cand.class_name} → {new_label}")

        # SAM refinement outcome.
        ref = next(
            (r for r in refine.results if r.candidate_id == cand.candidate_id),
            None,
        )
        if ref is not None:
            trace.append(
                f"refine: method={ref.method} "
                f"iou={ref.iou_with_original:.3f} "
                f"accepted={ref.accepted}"
            )

        return trace
