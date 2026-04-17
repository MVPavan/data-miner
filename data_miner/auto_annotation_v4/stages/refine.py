"""Stage 3: Refine — class-driven, per-prompt SAM extension + VLM adjudication.

Trigger (replaces the old VLM-signal-driven trigger):
    candidate enters refine iff
      class_name (post-relabel) in cfg.refine_rules.classes
      AND evaluate verdict in {accept, review}

Per-candidate flow (one candidate may run multiple prompts sequentially):
    for prompt in cfg.refine_rules.classes[class].prompts:
        VLM(refine_prompt) -> action in {skip, propose}
        skip                          -> next prompt
        propose target_region         -> SAM3 refine -> load_bbox
            -> SAM3 presence on load_bbox alone w/ load_vocab
                FAIL                  -> revert, next prompt
            -> merge_rules sanity on union(curr_bbox, load_bbox)
                FAIL                  -> revert, next prompt
            -> curr_bbox = merged
    final VLM adjudication (refine_adjudicate) -> accept | review | reject
    apply section 10.4 verdict combination (eval x refine_adjudicate)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import aiohttp
from PIL import Image
from pydantic import BaseModel

from ..workers.base import StageWorker
from ..configs import (
    AutoAnnotationV4Config,
    BoundingBox,
    Candidate,
    DetectorName,
    DetectorRequest,
    DetectorResponse,
    DetectResult,
    EvaluateResult,
    FinalAction,
    FinalAnnotation,
    MergeRulesConfig,
    PromptRef,
    PromptStepResult,
    RefineAction,
    RefineOutcome,
    RefineResult,
    RefinementInstruction,
    RefinementResult,
    SAM3RefineRequest,
    SAM3RefineResponse,
    Stage,
    StageMessage,
    Verdict,
    BboxSource,
    RefinePromptConfig,
)
from ..output import OutputWriter
from ..prompt_manager import load_prompt
from ..utils import (
    bbox_iou,
    bbox_to_pixels,
    crop_candidate,
    parse_vlm_json,
    pil_to_data_url,
)

logger = logging.getLogger("data_miner.auto_annotation_v4.refine")


# section 10.4 verdict combination table — keys are (evaluate_verdict, refine_adjudicate).
# Maps to (final_verdict, bbox_source). Note the key new rule:
#   accept + reject -> review (not accept original) — see section 10.4.
_COMBINATION_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    ("accept", "accept"): ("accept", "refined"),
    ("accept", "review"): ("review", "refined"),
    ("accept", "reject"): ("review", "original"),  # NEW per section 10.4
    ("review", "accept"): ("accept", "refined"),
    ("review", "review"): ("review", "refined"),
    ("review", "reject"): ("review", "original"),
}


class RefineWorker(StageWorker):
    """Stage 3 worker: class-driven SAM extension + VLM adjudication."""

    stage = Stage.REFINE

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        db: Any,
        *,
        output_writer: OutputWriter | None = None,
        server_semaphore: asyncio.Semaphore | None = None,
        worker_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(config, db, output_writer=output_writer, server_semaphore=server_semaphore, worker_id=worker_id, job_id=job_id)

    # ------------------------------------------------------------------
    # Core process
    # ------------------------------------------------------------------

    def _sam_port(self) -> int:
        """Port of whichever SAM3-family server is enabled.

        Prefers ``sam3_dart`` when both are enabled (it's strictly a drop-in
        replacement with the same wire). Raises if none are enabled — the
        refine stage has no fallback without SAM.
        """
        detectors = self.config.servers.detectors
        for name in (DetectorName.SAM3_DART, DetectorName.SAM3):
            cfg = detectors.get(name)
            if cfg is not None and cfg.enabled:
                return cfg.port
        raise RuntimeError(
            "Refine stage requires an enabled SAM3 server "
            "(sam3 or sam3_dart) in servers.detectors."
        )

    async def process(self, msg: StageMessage) -> BaseModel:
        t0 = time.monotonic()

        detect: DetectResult | None = await self.load_checkpoint(
            msg.image_id, Stage.DETECT, DetectResult
        )
        evaluate: EvaluateResult | None = await self.load_checkpoint(
            msg.image_id, Stage.EVALUATE, EvaluateResult
        )
        if detect is None:
            raise RuntimeError(f"No detect checkpoint for {msg.image_id}")

        # Evaluate may not exist if detect routed all auto-accepts directly to
        # refine (no VLM path). Treat as empty in that case.
        if evaluate is None:
            evaluate = EvaluateResult(
                image_id=msg.image_id,
                accepted=list(detect.routing.auto_accepted),
                review=[],
                rejected=[],
                relabels={},
            )

        image_w, image_h = detect.image_size[0], detect.image_size[1]
        sam_port = self._sam_port()
        refine_classes = self.config.refine_rules.classes

        # Build candidate lookup + identify those eligible for refine.
        cand_by_id: dict[str, Candidate] = {
            c.candidate_id: c for c in detect.candidates
        }
        accepted_set: set[str] = set(evaluate.accepted)
        review_set: set[str] = set(evaluate.review)

        eligible: list[tuple[Candidate, str]] = []  # (cand, eval_verdict)
        for cand in detect.candidates:
            cls_post = evaluate.relabels.get(cand.candidate_id, cand.class_name)
            if cls_post not in refine_classes:
                continue
            if cand.candidate_id in accepted_set:
                eligible.append((cand, "accept"))
            elif cand.candidate_id in review_set:
                eligible.append((cand, "review"))

        results: list[RefinementResult] = []
        instructions_used: list[RefinementInstruction] = []
        sam_calls = 0
        vlm_calls = 0
        prompt_used = PromptRef(prompt_id="refine_prompt", version="1", hash="")

        if not eligible:
            refine_result = self._build_refine_result(
                msg, results, instructions_used, sam_calls, vlm_calls,
                prompt_used, t0,
            )
            await self.save_checkpoint(msg.image_id, Stage.REFINE, refine_result)

            if self.output_writer is not None:
                self._write_final_output(msg.image_id, detect, evaluate, refine_result)

            return refine_result

        image = Image.open(msg.image_path).convert("RGB")

        for cand, eval_verdict in eligible:
            cls_post = evaluate.relabels.get(
                cand.candidate_id, cand.class_name
            )
            rule = refine_classes[cls_post]

            curr_bbox = cand.bbox
            steps: list[PromptStepResult] = []

            for prompt_cfg in rule.prompts:
                step, new_bbox, new_instructions, sc, vc = await self._run_prompt(
                    sam_port, msg.image_path, image,
                    cand, cls_post, curr_bbox, prompt_cfg, rule.merge_rules,
                    image_w, image_h,
                )
                sam_calls += sc
                vlm_calls += vc
                instructions_used.extend(new_instructions)
                steps.append(step)
                if new_bbox is not None:
                    curr_bbox = new_bbox

            bbox_changed = curr_bbox is not cand.bbox
            if bbox_changed:
                adjudicate, vc_adj = await self._adjudicate(
                    image, cand, cls_post, rule.prompts[0].description,
                    cand.bbox, curr_bbox, image_w, image_h,
                )
                vlm_calls += vc_adj
            else:
                # No prompt changed the bbox — trivially "accept original"
                # without consulting the adjudicator (saves a VLM call).
                adjudicate = "accept"

            final_verdict, bbox_source = self._combine(
                eval_verdict, adjudicate, bbox_changed
            )
            final_bbox = curr_bbox if bbox_source == "refined" else cand.bbox
            iou = bbox_iou(cand.bbox, final_bbox)

            results.append(
                RefinementResult(
                    candidate_id=cand.candidate_id,
                    original_bbox=cand.bbox,
                    refined_bbox=final_bbox if bbox_source == "refined" else None,
                    iou_with_original=iou,
                    accepted=(final_verdict == "accept"),
                    method="class_driven",
                    prompt_steps=steps,
                    adjudicate_verdict=Verdict(adjudicate),
                    final_verdict=Verdict(final_verdict),
                    final_bbox_source=BboxSource(bbox_source),
                )
            )

        refine_result = self._build_refine_result(
            msg, results, instructions_used, sam_calls, vlm_calls,
            prompt_used, t0,
        )
        await self.save_checkpoint(msg.image_id, Stage.REFINE, refine_result)

        self.logger.info(
            "%s: refine done — %d candidates, %d SAM, %d VLM, %.0f ms",
            msg.image_id, len(results), sam_calls, vlm_calls,
            (time.monotonic() - t0) * 1000,
        )

        if self.output_writer is not None:
            self._write_final_output(msg.image_id, detect, evaluate, refine_result)

        return refine_result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        return Stage.FINALIZE

    # ------------------------------------------------------------------
    # Build result helper
    # ------------------------------------------------------------------

    def _build_refine_result(
        self,
        msg: StageMessage,
        results: list[RefinementResult],
        instructions: list[RefinementInstruction],
        sam_calls: int,
        vlm_calls: int,
        prompt_used: PromptRef,
        t0: float,
    ) -> RefineResult:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return RefineResult(
            image_id=msg.image_id,
            refinement_instructions=instructions,
            results=results,
            vlm_calls=vlm_calls,
            sam_calls=sam_calls,
            prompt_used=prompt_used,
            stage_timing_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Per-prompt inner loop
    # ------------------------------------------------------------------

    async def _run_prompt(
        self,
        sam_port: int,
        image_path: str,
        image: Image.Image,
        cand: Candidate,
        class_name: str,
        curr_bbox: BoundingBox,
        prompt_cfg: RefinePromptConfig,
        merge_rules: MergeRulesConfig,
        image_w: int,
        image_h: int,
    ) -> tuple[
        PromptStepResult, BoundingBox | None, list[RefinementInstruction], int, int
    ]:
        """One pass of: VLM -> SAM -> presence -> merge sanity.

        Returns (step_trace, new_bbox_or_None, instructions_emitted, sam_calls, vlm_calls).
        new_bbox is None when the step did not produce a usable extension.
        """
        sam_calls = 0
        vlm_calls = 0
        instructions: list[RefinementInstruction] = []

        # ---- 1. VLM probe: skip vs propose ----
        try:
            instr = await self._vlm_propose(
                image, cand, class_name, curr_bbox, prompt_cfg,
                image_w, image_h,
            )
            vlm_calls += 1
            instructions.append(instr)
        except Exception as exc:
            self.logger.warning(
                "VLM propose failed for %s/%s: %s",
                cand.candidate_id, prompt_cfg.id, exc,
            )
            return (
                PromptStepResult(
                    prompt_id=prompt_cfg.id,
                    action=RefineAction.SKIP,
                    outcome=RefineOutcome.VLM_ERROR,
                    notes=str(exc)[:200],
                ),
                None, instructions, sam_calls, vlm_calls,
            )

        if instr.action == RefineAction.SKIP or instr.target_region is None:
            return (
                PromptStepResult(
                    prompt_id=prompt_cfg.id,
                    action=RefineAction.SKIP,
                    outcome=RefineOutcome.SKIPPED,
                    notes=instr.vlm_reasoning[:200],
                ),
                None, instructions, sam_calls, vlm_calls,
            )

        # ---- 2. SAM3 refine on target_region -> load bbox ----
        try:
            load_bbox = await self._sam_refine(
                sam_port, image_path, instr.target_region
            )
            sam_calls += 1
        except Exception as exc:
            self.logger.warning(
                "SAM refine failed for %s/%s: %s",
                cand.candidate_id, prompt_cfg.id, exc,
            )
            return (
                PromptStepResult(
                    prompt_id=prompt_cfg.id,
                    action=RefineAction.PROPOSE,
                    outcome=RefineOutcome.SAM_ERROR,
                    notes=str(exc)[:200],
                ),
                None, instructions, sam_calls, vlm_calls,
            )

        if load_bbox is None:
            return (
                PromptStepResult(
                    prompt_id=prompt_cfg.id,
                    action=RefineAction.PROPOSE,
                    outcome=RefineOutcome.SAM_NO_MASK,
                ),
                None, instructions, sam_calls, vlm_calls,
            )

        # ---- 3. Presence check on LOAD bbox alone (section 9.7) ----
        load_vocab = instr.load_vocab or prompt_cfg.load_vocab
        try:
            presence = await self._sam_presence(
                sam_port, image_path, load_bbox, load_vocab,
            )
            sam_calls += 1
        except Exception as exc:
            self.logger.warning(
                "Presence check failed for %s/%s: %s",
                cand.candidate_id, prompt_cfg.id, exc,
            )
            presence = 0.0

        if presence < prompt_cfg.presence_threshold:
            return (
                PromptStepResult(
                    prompt_id=prompt_cfg.id,
                    action=RefineAction.PROPOSE,
                    outcome=RefineOutcome.PRESENCE_FAILED,
                    presence_score=presence,
                    proposed_bbox=load_bbox,
                ),
                None, instructions, sam_calls, vlm_calls,
            )

        # ---- 4. Geometric merge sanity ----
        merged = _bbox_union(curr_bbox, load_bbox)
        if not _passes_merge_rules(curr_bbox, load_bbox, merged, merge_rules):
            return (
                PromptStepResult(
                    prompt_id=prompt_cfg.id,
                    action=RefineAction.PROPOSE,
                    outcome=RefineOutcome.MERGE_FAILED,
                    presence_score=presence,
                    proposed_bbox=load_bbox,
                    merged_bbox=merged,
                ),
                None, instructions, sam_calls, vlm_calls,
            )

        return (
            PromptStepResult(
                prompt_id=prompt_cfg.id,
                action=RefineAction.PROPOSE,
                outcome=RefineOutcome.MERGED,
                presence_score=presence,
                proposed_bbox=load_bbox,
                merged_bbox=merged,
            ),
            merged, instructions, sam_calls, vlm_calls,
        )

    # ------------------------------------------------------------------
    # VLM helpers
    # ------------------------------------------------------------------

    async def _vlm_propose(
        self,
        image: Image.Image,
        cand: Candidate,
        class_name: str,
        curr_bbox: BoundingBox,
        prompt_cfg: RefinePromptConfig,
        image_w: int,
        image_h: int,
    ) -> RefinementInstruction:
        template = load_prompt("refine_prompt")
        px1, py1, px2, py2 = bbox_to_pixels(curr_bbox, image_w, image_h)
        rendered, _ = template.render_and_hash(
            class_name=class_name,
            image_width=image_w,
            image_height=image_h,
            x1=px1, y1=py1, x2=px2, y2=py2,
            description=prompt_cfg.description.strip(),
        )

        crop = crop_candidate(image, curr_bbox, padding=0.20)
        messages = [
            {"role": "system", "content": rendered},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_to_data_url(crop)}},
                    {"type": "text", "text": "Decide skip or propose per the rule."},
                ],
            },
        ]

        data = await self._vlm_call(messages, template, max_tokens=384)
        action = str(data.get("action", "skip")).lower()
        target_region = data.get("target_region")
        load_vocab = data.get("load_vocab") or []
        reasoning = str(data.get("reasoning", ""))

        target_bbox: BoundingBox | None = None
        if action == "propose" and isinstance(target_region, dict):
            try:
                target_bbox = BoundingBox(
                    x1=float(target_region["x1"]) / image_w,
                    y1=float(target_region["y1"]) / image_h,
                    x2=float(target_region["x2"]) / image_w,
                    y2=float(target_region["y2"]) / image_h,
                )
            except (KeyError, TypeError, ValueError):
                target_bbox = None

        if target_bbox is None:
            action = "skip"

        # Normalise action to the StrEnum
        refine_action = RefineAction.PROPOSE if action == "propose" else RefineAction.SKIP

        return RefinementInstruction(
            candidate_id=cand.candidate_id,
            prompt_id=prompt_cfg.id,
            action=refine_action,
            target_region=target_bbox,
            load_vocab=[str(v) for v in load_vocab if isinstance(v, str)],
            vlm_reasoning=reasoning,
        )

    async def _adjudicate(
        self,
        image: Image.Image,
        cand: Candidate,
        class_name: str,
        rule_description: str,
        original: BoundingBox,
        refined: BoundingBox,
        image_w: int,
        image_h: int,
    ) -> tuple[str, int]:
        """Final VLM adjudication on refined-vs-original. Returns (verdict, vlm_calls)."""
        template = load_prompt("refine_adjudicate")
        ox1, oy1, ox2, oy2 = bbox_to_pixels(original, image_w, image_h)
        rx1, ry1, rx2, ry2 = bbox_to_pixels(refined, image_w, image_h)
        rendered, _ = template.render_and_hash(
            class_name=class_name,
            description=rule_description.strip(),
            image_width=image_w,
            image_height=image_h,
            orig_x1=ox1, orig_y1=oy1, orig_x2=ox2, orig_y2=oy2,
            ref_x1=rx1, ref_y1=ry1, ref_x2=rx2, ref_y2=ry2,
        )

        # Show the refined crop with extra padding so adjudicator sees context.
        crop = crop_candidate(image, refined, padding=0.10)
        messages = [
            {"role": "system", "content": rendered},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_to_data_url(crop)}},
                    {"type": "text", "text": "Adjudicate per the rule."},
                ],
            },
        ]
        try:
            data = await self._vlm_call(messages, template, max_tokens=256)
        except Exception as exc:
            self.logger.warning(
                "Adjudicate failed for %s: %s — defaulting to review",
                cand.candidate_id, exc,
            )
            return "review", 1

        verdict = str(data.get("verdict", "review")).lower()
        if verdict not in ("accept", "review", "reject"):
            verdict = "review"
        return verdict, 1

    async def _vlm_call(
        self,
        messages: list[dict],
        template: Any,
        max_tokens: int,
    ) -> dict:
        vlm_cfg = self.config.servers.vlm
        payload = {
            "model": vlm_cfg.model,
            "messages": messages,
            "temperature": template.model_params.get(
                "temperature", vlm_cfg.temperature
            ),
            "max_tokens": template.model_params.get("max_tokens", max_tokens),
        }
        async with asyncio.timeout(60):
            async with self._session.post(
                f"{vlm_cfg.url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {vlm_cfg.api_key}"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                response = await resp.json()
        text = response["choices"][0]["message"]["content"]
        parsed = parse_vlm_json(text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        return parsed if isinstance(parsed, dict) else {}

    # ------------------------------------------------------------------
    # SAM3 helpers (normalized bbox throughout)
    # ------------------------------------------------------------------

    async def _sam_refine(
        self,
        sam_port: int,
        image_path: str,
        bbox: BoundingBox,
    ) -> BoundingBox | None:
        url = f"http://localhost:{sam_port}/predict"
        req = SAM3RefineRequest(
            image_path=str(image_path),
            bbox=[bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        )
        acquired = False
        try:
            if self._server_semaphore is not None:
                await self._server_semaphore.acquire()
                acquired = True
            async with asyncio.timeout(30):
                async with self._session.post(
                    url, json=req.model_dump(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except TimeoutError:
            self.logger.warning("SAM refine timed out for bbox")
            return None
        finally:
            if acquired:
                self._server_semaphore.release()
        resp_model = SAM3RefineResponse.model_validate(data)
        if resp_model.box is None or len(resp_model.box) != 4:
            return None
        return BoundingBox(
            x1=resp_model.box[0], y1=resp_model.box[1],
            x2=resp_model.box[2], y2=resp_model.box[3],
        )

    async def _sam_presence(
        self,
        sam_port: int,
        image_path: str,
        load_bbox: BoundingBox,
        load_vocab: list[str],
    ) -> float:
        """Query SAM3 presence head with load_vocab; return max score over
        proposals that overlap ``load_bbox`` by IoU >= 0.3.

        Falls back to 0.0 when no vocab supplied.
        """
        if not load_vocab:
            return 0.0
        prompts = [v.strip() for v in load_vocab if v.strip()]
        if not prompts:
            return 0.0

        url = f"http://localhost:{sam_port}/predict"
        req = DetectorRequest(
            image_path=str(image_path), prompts=prompts, threshold=0.0,
        )
        acquired = False
        try:
            if self._server_semaphore is not None:
                await self._server_semaphore.acquire()
                acquired = True
            async with asyncio.timeout(30):
                async with self._session.post(
                    url, json=req.model_dump(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except TimeoutError:
            self.logger.warning("SAM presence timed out for bbox")
            return 0.0
        finally:
            if acquired:
                self._server_semaphore.release()

        resp_model = DetectorResponse.model_validate(data)
        best = 0.0
        for b, s in zip(resp_model.boxes, resp_model.scores):
            try:
                cand_box = BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])
            except Exception:
                continue
            if bbox_iou(load_bbox, cand_box) >= 0.3 and float(s) > best:
                best = float(s)
        return best

    # ------------------------------------------------------------------
    # Verdict combination + output
    # ------------------------------------------------------------------

    @staticmethod
    def _combine(
        eval_verdict: str,
        adjudicate: str,
        bbox_changed: bool,
    ) -> tuple[str, str]:
        if not bbox_changed:
            # No extension produced — final = eval_verdict, bbox = original.
            return eval_verdict, "original"
        return _COMBINATION_TABLE.get(
            (eval_verdict, adjudicate), ("review", "original")
        )

    # ------------------------------------------------------------------
    # Final output (legacy path — finalize stage is the canonical sink,
    # but refine still writes when output_writer is set for convenience.)
    # ------------------------------------------------------------------

    def _write_final_output(
        self,
        image_id: str,
        detect: DetectResult,
        evaluate: EvaluateResult,
        refine: RefineResult,
    ) -> None:
        # v4: config.classes is dict[str, ClassConfig]
        class_map: dict[str, int] = {
            name: cfg.id for name, cfg in self.config.classes.items()
        }
        refine_by_id: dict[str, RefinementResult] = {
            r.candidate_id: r for r in refine.results
        }
        accepted_set: set[str] = set(evaluate.accepted)
        review_set: set[str] = set(evaluate.review)
        rejected_set: set[str] = set(evaluate.rejected)

        annotations: list[FinalAnnotation] = []
        review_items: list[dict] = []

        for cand in detect.candidates:
            if cand.candidate_id in rejected_set:
                continue
            cls_post = evaluate.relabels.get(cand.candidate_id, cand.class_name)
            class_id = class_map.get(cls_post, -1)

            ref = refine_by_id.get(cand.candidate_id)
            if ref is not None:
                bbox = (
                    ref.refined_bbox
                    if ref.final_bbox_source == BboxSource.REFINED and ref.refined_bbox is not None
                    else ref.original_bbox
                )
                if ref.final_verdict == Verdict.ACCEPT:
                    annotations.append(FinalAnnotation(
                        candidate_id=cand.candidate_id,
                        class_name=cls_post, class_id=class_id, bbox=bbox,
                        confidence=cand.score, action=FinalAction.ACCEPT,
                        source_model=cand.source_model,
                        was_refined=(ref.final_bbox_source == BboxSource.REFINED),
                        trace=self._build_trace(cand, evaluate, ref),
                    ))
                elif ref.final_verdict == Verdict.REVIEW:
                    review_items.append({
                        "candidate_id": cand.candidate_id,
                        "class_name": cls_post,
                        "bbox": bbox.model_dump(),
                        "reason": "refine_review",
                        "adjudicate_verdict": ref.adjudicate_verdict,
                        "iou_with_original": ref.iou_with_original,
                    })
                # ref.final_verdict == "reject" -> drop entirely.
            else:
                # Not refined (class not in refine_rules, or not eligible).
                if cand.candidate_id in accepted_set:
                    annotations.append(FinalAnnotation(
                        candidate_id=cand.candidate_id,
                        class_name=cls_post, class_id=class_id, bbox=cand.bbox,
                        confidence=cand.score, action=FinalAction.ACCEPT,
                        source_model=cand.source_model, was_refined=False,
                        trace=["accepted by VLM evaluate"],
                    ))
                elif cand.candidate_id in review_set:
                    review_items.append({
                        "candidate_id": cand.candidate_id,
                        "class_name": cls_post,
                        "bbox": cand.bbox.model_dump(),
                        "reason": "evaluate_review",
                    })

        self.output_writer.write_yolo_labels(image_id, annotations, class_map)
        self.output_writer.write_trace(image_id, {
            "image_id": image_id,
            "stages": ["detect", "evaluate", "refine"],
            "detect": detect.model_dump(mode="json"),
            "evaluate": evaluate.model_dump(mode="json"),
            "refine": refine.model_dump(mode="json"),
            "annotations": [a.model_dump(mode="json") for a in annotations],
        })
        if review_items:
            self.output_writer.write_review(image_id, review_items)

        self.logger.info(
            "%s: wrote %d annotations, %d review items",
            image_id, len(annotations), len(review_items),
        )

    @staticmethod
    def _build_trace(
        cand: Candidate, evaluate: EvaluateResult, ref: RefinementResult
    ) -> list[str]:
        trace: list[str] = []
        verdict = next(
            (v for v in evaluate.verdicts if v.candidate_id == cand.candidate_id),
            None,
        )
        if verdict is not None:
            trace.append(
                f"vlm: class={verdict.correct_class} conf={verdict.confidence:.2f}"
            )
        relabel = evaluate.relabels.get(cand.candidate_id)
        if relabel and relabel != cand.class_name:
            trace.append(f"relabel: {cand.class_name} -> {relabel}")
        for s in ref.prompt_steps:
            extra = ""
            if s.presence_score is not None:
                extra = f" presence={s.presence_score:.2f}"
            trace.append(f"refine[{s.prompt_id}]: {s.outcome}{extra}")
        trace.append(
            f"adjudicate={ref.adjudicate_verdict} final={ref.final_verdict} "
            f"bbox={ref.final_bbox_source}"
        )
        return trace


# ---------------------------------------------------------------------------
# Pure helpers (not on the worker class)
# ---------------------------------------------------------------------------


def _bbox_union(a: BoundingBox, b: BoundingBox) -> BoundingBox:
    return BoundingBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def _passes_merge_rules(
    orig: BoundingBox,
    load: BoundingBox,
    merged: BoundingBox,
    rules: MergeRulesConfig,
) -> bool:
    """Geometric sanity gates per section 9.6 step 4."""
    # area cap
    orig_area = max(orig.area, 1e-9)
    if merged.area / orig_area > rules.max_area_ratio:
        return False
    # gap (only enforce when not overlapping)
    if bbox_iou(orig, load) <= 0:
        diag = math.sqrt(1.0 ** 2 + 1.0 ** 2)  # normalized image diag
        # closest-edge gap in normalized units
        dx = max(0.0, max(load.x1 - orig.x2, orig.x1 - load.x2))
        dy = max(0.0, max(load.y1 - orig.y2, orig.y1 - load.y2))
        gap = math.sqrt(dx * dx + dy * dy)
        if gap > rules.max_gap_diag_frac * diag:
            return False
    # aspect
    ar = merged.aspect_ratio
    lo, hi = rules.aspect_ratio_range
    if ar < lo or ar > hi:
        return False
    return True
