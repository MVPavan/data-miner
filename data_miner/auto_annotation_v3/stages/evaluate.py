"""Stage 2: Evaluate — VLM classification + quality + spatial refinement instructions."""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp
from PIL import Image

from ..contracts import (
    BboxQuality,
    Candidate,
    DetectResult,
    EvaluateResult,
    FinalAction,
    FinalAnnotation,
    PromptRef,
    StageMessage,
    VLMVerdict,
)
from ..prompt_manager import load_prompt
from ..utils import (
    bbox_to_pixels,
    build_class_alias_map,
    crop_candidate,
    draw_candidates_on_image,
    parse_vlm_json,
    pil_to_data_url,
    resolve_canonical_class,
)
from ..workers.base import StageWorker

logger = logging.getLogger("data_miner.auto_annotation_v3.evaluate")


class EvaluateWorker(StageWorker):
    """Stage 2 worker: VLM classification + quality assessment + spatial refinement."""

    stage = "evaluate"

    def __init__(self, config, broker, checkpoint_mgr, output_writer=None, **kwargs):
        super().__init__(config, broker, checkpoint_mgr, **kwargs)
        self.output_writer = output_writer
        self.alias_map = build_class_alias_map(self.config.classes)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> StageMessage | None:
        t0 = time.monotonic()

        # 1. Load detect checkpoint — mandatory prerequisite.
        detect: DetectResult | None = self.load_checkpoint(
            msg.image_id, "detect", DetectResult
        )
        if detect is None:
            raise RuntimeError(f"No detect checkpoint for {msg.image_id}")

        # 2. Identify candidates that need VLM evaluation.
        to_eval: list[Candidate] = [
            c
            for c in detect.candidates
            if c.candidate_id in detect.routing.needs_evaluation
        ]

        if not to_eval:
            # All candidates were auto-accepted — skip VLM entirely.
            eval_result = EvaluateResult(
                image_id=msg.image_id,
                vlm_calls=0,
                vlm_total_tokens=0,
                prompts_used=[],
                verdicts=[],
                accepted=list(detect.routing.auto_accepted),
                review=[],
                rejected=[],
                relabels={},
                stage_timing_ms=(time.monotonic() - t0) * 1000,
            )
            self.save_checkpoint(msg.image_id, "evaluate", eval_result)
            return self._route_after_evaluate(msg, detect, eval_result)

        # 3. Open image once; share across all VLM calls.
        image = Image.open(msg.image_path).convert("RGB")
        image_w, image_h = image.size

        vlm_calls = 0
        vlm_total_tokens = 0
        prompts_used: list[PromptRef] = []
        all_verdicts: list[VLMVerdict] = []

        # 4. Group candidates by evaluation group, then classify in parallel.
        groups = self._group_by_eval_group(to_eval)

        vlm_cfg = self.config.servers.vlm
        timeout = aiohttp.ClientTimeout(total=vlm_cfg.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:

            # ---- VLM Call 1: Classification + Quality (one call per group) ----
            group_tasks = [
                self._classify_group(session, msg, detect, image, group_name, candidates)
                for group_name, candidates in groups.items()
                if candidates
            ]
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

            for result in group_results:
                if isinstance(result, Exception):
                    self.logger.error(
                        "Group classification failed: %s", result, exc_info=result
                    )
                    continue
                vlm_calls += 1
                vlm_total_tokens += result["tokens"]
                prompts_used.append(
                    PromptRef(
                        group=result["group"],
                        prompt_id=result["prompt_id"],
                        version=result["prompt_version"],
                        hash=result["prompt_hash"],
                    )
                )
                all_verdicts.extend(result["verdicts"])

            # ---- Resolve verdicts to three-way routing (accept / review / reject) ----
            accepted, review, rejected, relabels = self._resolve_verdicts(
                all_verdicts, detect
            )

        # Merge auto-accepted candidates from detect with VLM-accepted ones.
        all_accepted = list(detect.routing.auto_accepted) + accepted

        elapsed_ms = (time.monotonic() - t0) * 1000
        eval_result = EvaluateResult(
            image_id=msg.image_id,
            vlm_calls=vlm_calls,
            vlm_total_tokens=vlm_total_tokens,
            prompts_used=prompts_used,
            verdicts=all_verdicts,
            accepted=all_accepted,
            review=review,
            rejected=rejected,
            relabels=relabels,
            stage_timing_ms=elapsed_ms,
        )
        self.save_checkpoint(msg.image_id, "evaluate", eval_result)

        return self._route_after_evaluate(msg, detect, eval_result)

    # ------------------------------------------------------------------
    # Post-evaluate routing — class-driven refine trigger
    # ------------------------------------------------------------------

    def _route_after_evaluate(
        self,
        msg: StageMessage,
        detect: DetectResult,
        eval_result: EvaluateResult,
    ) -> StageMessage | None:
        """Forward to refine if any non-rejected survivor's class is in
        refine_rules; otherwise write final output and forward to done.
        """
        refine_classes = set(self.config.refine_rules.classes.keys())
        if refine_classes:
            survivors_post_relabel = []
            for cand in detect.candidates:
                if cand.candidate_id in eval_result.rejected:
                    continue
                if cand.candidate_id not in (
                    set(eval_result.accepted) | set(eval_result.review)
                ):
                    continue
                cls = eval_result.relabels.get(cand.candidate_id, cand.class_name)
                if cls in refine_classes:
                    survivors_post_relabel.append(cand.candidate_id)
            if survivors_post_relabel:
                return msg.forward("refine")

        return msg.forward("finalize")

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def _group_by_eval_group(
        self, candidates: list[Candidate]
    ) -> dict[str, list[Candidate]]:
        """Map candidates to their configured evaluation group."""
        class_to_group: dict[str, str] = {}
        for group_name, group_cfg in self.config.active_evaluation_groups.items():
            for cls_name in group_cfg.classes:
                class_to_group[cls_name] = group_name

        groups: dict[str, list[Candidate]] = {}
        for cand in candidates:
            group = class_to_group.get(cand.class_name, "default")
            groups.setdefault(group, []).append(cand)

        return groups

    # ------------------------------------------------------------------
    # VLM Call 1: Classification + Quality
    # ------------------------------------------------------------------

    async def _classify_group(
        self,
        session: aiohttp.ClientSession,
        msg: StageMessage,
        detect: DetectResult,
        image: Image.Image,
        group_name: str,
        candidates: list[Candidate],
    ) -> dict:
        """One VLM call for a group of candidates: returns classification verdicts."""
        template = load_prompt(f"classify_{group_name}")
        group_cfg = self.config.active_evaluation_groups.get(group_name)

        class_list = ", ".join(group_cfg.classes) if group_cfg else ""
        class_descriptions = (group_cfg.description or "") if group_cfg else ""
        annotation_rules = self._format_rules(group_cfg) if group_cfg else ""

        rendered, prompt_hash = template.render_and_hash(
            class_list=class_list,
            class_descriptions=class_descriptions,
            annotation_rules=annotation_rules,
        )

        # Build image content list: annotated overview + optional per-candidate crops.
        images_content: list[dict] = []

        annotated = draw_candidates_on_image(image.copy(), candidates)
        images_content.append(
            {"type": "image_url", "image_url": {"url": pil_to_data_url(annotated)}}
        )

        if group_cfg and group_cfg.requires_crops:
            for cand in candidates:
                crop = crop_candidate(image, cand.bbox)
                images_content.append(
                    {"type": "image_url", "image_url": {"url": pil_to_data_url(crop)}}
                )

        messages = [
            {"role": "system", "content": rendered},
            {
                "role": "user",
                "content": images_content
                + [
                    {
                        "type": "text",
                        "text": f"Evaluate these {len(candidates)} candidates.",
                    }
                ],
            },
        ]

        vlm_cfg = self.config.servers.vlm
        payload = {
            "model": vlm_cfg.model,
            "messages": messages,
            "temperature": template.model_params.get("temperature", vlm_cfg.temperature),
            "max_tokens": template.model_params.get("max_tokens", vlm_cfg.max_tokens),
        }

        vlm_url = f"{vlm_cfg.url}/chat/completions"
        async with session.post(
            vlm_url,
            json=payload,
            headers={"Authorization": f"Bearer {vlm_cfg.api_key}"},
        ) as resp:
            resp.raise_for_status()
            vlm_response = await resp.json()

        response_text = vlm_response["choices"][0]["message"]["content"]
        verdicts_raw = parse_vlm_json(response_text)

        verdicts: list[VLMVerdict] = []
        if isinstance(verdicts_raw, list):
            for v in verdicts_raw:
                try:
                    verdicts.append(
                        VLMVerdict(
                            candidate_id=str(v.get("candidate_id", "")),
                            correct_class=str(v.get("correct_class", "")),
                            confidence=float(v.get("confidence", 0.0)),
                            bbox_quality=BboxQuality(
                                v.get("bbox_quality", BboxQuality.GOOD)
                            ),
                            object_complete=bool(v.get("object_complete", True)),
                            reasoning=str(v.get("reasoning", "")),
                        )
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Skipping malformed verdict entry for group '%s': %s — %s",
                        group_name,
                        v,
                        exc,
                    )

        verdicts = self._map_verdict_ids(verdicts, candidates)

        tokens: int = vlm_response.get("usage", {}).get("total_tokens", 0)

        return {
            "group": group_name,
            "prompt_id": template.id,
            "prompt_version": template.version,
            "prompt_hash": prompt_hash,
            "verdicts": verdicts,
            "tokens": tokens,
        }

    def _map_verdict_ids(
        self, verdicts: list[VLMVerdict], candidates: list[Candidate]
    ) -> list[VLMVerdict]:
        """Remap VLM-returned IDs (numeric or partial) to real candidate_ids."""
        cand_id_set = {c.candidate_id for c in candidates}
        mapped: list[VLMVerdict] = []
        for v in verdicts:
            if v.candidate_id in cand_id_set:
                mapped.append(v)
                continue
            # Attempt numeric-index interpretation (VLM may return "0", "1", …).
            try:
                idx = int(v.candidate_id)
                if 0 <= idx < len(candidates):
                    # VLMVerdict has model_config extra="forbid", so mutate in place
                    # by replacing the model instance.
                    mapped.append(
                        v.model_copy(
                            update={"candidate_id": candidates[idx].candidate_id}
                        )
                    )
                    continue
            except (ValueError, TypeError):
                pass
            self.logger.warning(
                "Could not map verdict candidate_id %r to any known candidate",
                v.candidate_id,
            )
        return mapped

    # ------------------------------------------------------------------
    # Verdict resolution
    # ------------------------------------------------------------------

    def _resolve_verdicts(
        self,
        verdicts: list[VLMVerdict],
        detect: DetectResult,
    ) -> tuple[list[str], list[str], list[str], dict[str, str]]:
        """Partition verdict IDs into accept / review / reject buckets.

        Pure confidence-driven three-way (§10.2). ``bbox_quality`` and
        ``object_complete`` remain on the verdict for telemetry but no longer
        drive routing — spatial decisions belong to the refine stage.

        Returns
        -------
        accepted, review, rejected, relabels
        """
        accepted: list[str] = []
        review: list[str] = []
        rejected: list[str] = []
        relabels: dict[str, str] = {}

        cand_by_id: dict[str, Candidate] = {
            c.candidate_id: c for c in detect.candidates
        }
        eval_cfg = self.config.evaluate

        for v in verdicts:
            cand = cand_by_id.get(v.candidate_id)

            # ---- Relabeling / rejection by class ----
            original_class = cand.class_name if cand else ""
            if v.correct_class and v.correct_class != original_class:
                if v.correct_class.lower() in ("other", "unknown", "none"):
                    rejected.append(v.candidate_id)
                    continue
                canonical = resolve_canonical_class(v.correct_class, self.alias_map)
                if canonical is not None and canonical != original_class:
                    relabels[v.candidate_id] = canonical
                # Unresolvable → fall through to confidence routing.

            # ---- Confidence-driven three-way routing ----
            if v.confidence < eval_cfg.reject_below:
                rejected.append(v.candidate_id)
            elif v.confidence >= eval_cfg.accept_above:
                accepted.append(v.candidate_id)
            else:
                review.append(v.candidate_id)

        return accepted, review, rejected, relabels

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _format_rules(self, group_cfg) -> str:
        """Format per-class annotation rules into a readable string."""
        if not group_cfg or not group_cfg.annotation_rules:
            return ""
        return "\n".join(
            f"- {cls_name.upper()}: {rule}"
            for cls_name, rule in group_cfg.annotation_rules.items()
        )

    def _write_final_output(
        self, image_id: str, detect: DetectResult, evaluate: EvaluateResult
    ) -> None:
        """Write YOLO labels, audit trace, and review items when no refinement needed."""
        class_map: dict[str, int] = {c.name: c.id for c in self.config.classes}

        annotations: list[FinalAnnotation] = []
        for cand in detect.candidates:
            if cand.candidate_id not in evaluate.accepted:
                continue
            cls_name = evaluate.relabels.get(cand.candidate_id, cand.class_name)
            annotations.append(
                FinalAnnotation(
                    candidate_id=cand.candidate_id,
                    class_name=cls_name,
                    class_id=class_map.get(cls_name, -1),
                    bbox=cand.bbox,
                    confidence=cand.score,
                    action=FinalAction.ACCEPT,
                    source_model=cand.source_model,
                    was_refined=False,
                    trace=["accepted by VLM evaluate"],
                )
            )

        self.output_writer.write_yolo_labels(image_id, annotations, class_map)

        self.output_writer.write_trace(
            image_id,
            {
                "image_id": image_id,
                "stages": ["detect", "evaluate"],
                "detect": detect.model_dump(mode="json"),
                "evaluate": evaluate.model_dump(mode="json"),
                "annotations": [a.model_dump(mode="json") for a in annotations],
            },
        )

        # Candidates routed to `review` (uncertain confidence) go to the
        # human-review queue. Rejected candidates are dropped silently —
        # finalize stage (Phase 3) is the canonical place for drop logging.
        review_candidates = [
            cand
            for cand in detect.candidates
            if cand.candidate_id in evaluate.review
        ]
        if review_candidates:
            self.output_writer.write_review(image_id, review_candidates)
