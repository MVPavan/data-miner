"""Stage 2: Evaluate — VLM classification + quality + spatial refinement instructions."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp
from PIL import Image
from pydantic import BaseModel

from ..configs import (
    AutoAnnotationV4Config,
    BboxQuality,
    Candidate,
    DetectResult,
    EvaluateResult,
    FilterContext,
    FilterDrop,
    FilterResult,
    FinalAction,
    FinalAnnotation,
    PromptRef,
    Stage,
    StageMessage,
    VLMVerdict,
)
from ..filters import FilterPipeline
from ..prompt_manager import load_prompt
from ..utils import (
    crop_candidate,
    draw_focus_on_image,
    parse_vlm_json,
    pil_to_data_url,
    resolve_canonical_class,
)
from ..workers.base import StageWorker
from ..workers.http_retry import http_retry
from .detect import _build_v4_alias_map

logger = logging.getLogger("data_miner.auto_annotation_v4.evaluate")


class EvaluateWorker(StageWorker):
    """Stage 2 worker: VLM classification + quality assessment + spatial refinement."""

    stage = Stage.EVALUATE

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        db: Any,
        *,
        output_writer: Any | None = None,
        worker_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(config, db, output_writer=output_writer, worker_id=worker_id, job_id=job_id)
        self.alias_map = _build_v4_alias_map(self.config.classes)
        # Cache a single FilterPipeline per worker — constructor is cheap but
        # we still build it once so POST_REVIEW calls share one instance.
        self._filter_pipeline = FilterPipeline(self.config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> BaseModel:
        t0 = time.monotonic()

        # 1. Load filter checkpoint — mandatory prerequisite (Phase 2b: the
        # filter stage owns POST_DETECT filtering + routing, so evaluate reads
        # from FILTER, not DETECT).
        filter_result: FilterResult | None = await self.load_checkpoint(
            msg.image_id, Stage.FILTER, FilterResult
        )
        if filter_result is None:
            raise RuntimeError(f"filter stage missing for {msg.image_id}")

        candidates_in = filter_result.candidates
        routing = filter_result.routing

        # Stash candidates + routing for _resolve_next_stage and POST_REVIEW
        # rebuild (which needs original class names + bboxes).
        self._last_candidates = candidates_in
        self._last_routing = routing

        # 2. Identify candidates that need VLM evaluation.
        to_eval: list[Candidate] = [
            c
            for c in candidates_in
            if c.candidate_id in routing.needs_evaluation
        ]

        if not to_eval:
            # All candidates were auto-accepted — skip VLM entirely.
            eval_result = EvaluateResult(
                image_id=msg.image_id,
                vlm_calls=0,
                vlm_total_tokens=0,
                prompts_used=[],
                verdicts=[],
                accepted=list(routing.auto_accepted),
                review=[],
                rejected=[],
                relabels={},
                drops=[],
                stage_timing_ms=(time.monotonic() - t0) * 1000,
            )
            await self.save_checkpoint(msg.image_id, Stage.EVALUATE, eval_result)
            return eval_result

        # 3. Open image once; share across all VLM calls.
        image = Image.open(msg.image_path).convert("RGB")
        image_w, image_h = image.size

        vlm_calls = 0
        vlm_total_tokens = 0
        prompts_used: list[PromptRef] = []
        all_verdicts: list[VLMVerdict] = []

        # 4. Per-candidate concurrent VLM classification.
        # Group lookup is still used to inject the group's class context (class
        # list, descriptions, annotation rules) into each per-candidate prompt —
        # but the request itself carries only that one candidate's crop.
        class_to_group = self._build_class_to_group()

        sem = asyncio.Semaphore(max(1, self.config.evaluate.concurrency))

        tasks = [
            self._classify_one(sem, image, cand, class_to_group)
            for cand in to_eval
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_prompts: set[tuple[str, str]] = set()
        for cand, result in zip(to_eval, results):
            if isinstance(result, Exception):
                self.logger.warning(
                    "Per-candidate classify failed for %s: %s",
                    cand.candidate_id, result,
                )
                continue
            if result is None:
                continue
            vlm_calls += 1
            vlm_total_tokens += result["tokens"]
            key = (result["group"], result["prompt_hash"])
            if key not in seen_prompts:
                seen_prompts.add(key)
                prompts_used.append(PromptRef(
                    group=result["group"],
                    prompt_id=result["prompt_id"],
                    version=result["prompt_version"],
                    hash=result["prompt_hash"],
                ))
            all_verdicts.append(result["verdict"])

        # ---- Resolve verdicts to three-way routing (accept / review / reject) ----
        accepted, review, rejected, relabels = self._resolve_verdicts(
            all_verdicts, candidates_in
        )

        # Merge auto-accepted candidates from filter with VLM-accepted ones.
        all_accepted = list(routing.auto_accepted) + accepted

        # ---- POST_REVIEW filter pass (cross_class + per_class_cap) ----
        # After VLM verdicts + relabels, re-check class-level invariants that
        # relabels may have broken. Only survivors (accept | review) go through
        # the filter; rejected candidates are excluded outright.
        survivor_ids: set[str] = set(all_accepted) | set(review)
        candidates_after_verdicts: list[Candidate] = []
        for cand in candidates_in:
            if cand.candidate_id not in survivor_ids:
                continue
            cls_post = relabels.get(cand.candidate_id, cand.class_name)
            if cls_post == cand.class_name:
                candidates_after_verdicts.append(cand)
            else:
                # Re-labeled survivor — clone with updated class_name so filters
                # see the post-VLM class for cross-class + per-class-cap rules.
                candidates_after_verdicts.append(cand.model_copy(update={"class_name": cls_post}))

        kept, drops = self._filter_pipeline.run(
            candidates_after_verdicts, FilterContext.POST_REVIEW
        )
        dropped_ids: set[str] = {d.candidate_id for d in drops}
        # Remove filter-dropped ids from accepted / review buckets.
        all_accepted = [cid for cid in all_accepted if cid not in dropped_ids]
        review = [cid for cid in review if cid not in dropped_ids]

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
            drops=drops,
            stage_timing_ms=elapsed_ms,
        )
        await self.save_checkpoint(msg.image_id, Stage.EVALUATE, eval_result)

        return eval_result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_stage(self, result: BaseModel) -> Stage:
        """Forward to refine if any non-rejected survivor's class (post-relabel)
        is in refine_rules; otherwise forward to finalize.

        Uses ``self._last_candidates`` (stashed during :meth:`process`) to
        resolve original class names for non-relabeled candidates — mirrors v3
        ``_route_after_evaluate`` exactly.
        """
        eval_result: EvaluateResult = result  # type: ignore[assignment]
        refine_classes = set(self.config.refine_rules.classes.keys())
        if not refine_classes:
            return Stage.FINALIZE

        candidates: list[Candidate] | None = getattr(self, "_last_candidates", None)
        if candidates is None:
            # Should never happen — process() always sets _last_candidates.
            return Stage.FINALIZE

        survivor_ids = set(eval_result.accepted) | set(eval_result.review)
        for cand in candidates:
            if cand.candidate_id in eval_result.rejected:
                continue
            if cand.candidate_id not in survivor_ids:
                continue
            cls = eval_result.relabels.get(cand.candidate_id, cand.class_name)
            if cls in refine_classes:
                return Stage.REFINE

        return Stage.FINALIZE

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def _build_class_to_group(self) -> dict[str, str]:
        """class_name -> evaluation_group_name (used to inject per-group context
        into each per-candidate VLM prompt)."""
        out: dict[str, str] = {}
        for group_name, group_cfg in self.config.active_evaluation_groups.items():
            for cls_name in group_cfg.classes:
                out[cls_name] = group_name
        return out

    # ------------------------------------------------------------------
    # Per-candidate VLM call (replaces the old grouped multi-image call --
    # one HTTP request per candidate, fired concurrently and bounded by an
    # asyncio.Semaphore. vLLM's continuous batcher merges the in-flight
    # requests into one forward pass, so wall-clock cost is similar while
    # each prompt stays small enough to fit max_model_len.)
    # ------------------------------------------------------------------

    async def _classify_one(
        self,
        sem: asyncio.Semaphore,
        image: Image.Image,
        cand: Candidate,
        class_to_group: dict[str, str],
    ) -> dict | None:
        """One VLM call for one candidate. Returns dict or None on failure.

        Uses ``self._session`` (the shared aiohttp session created by
        :class:`StageWorker`).

        Always sends a bbox-highlighted overview. Sends an additional close-up
        crop when the candidate's evaluation group has ``requires_crops: true``.
        Picks the matching prompt template (`classify_one` vs
        `classify_one_with_crop`).
        """
        group_name = class_to_group.get(cand.class_name, "default")
        group_cfg = self.config.active_evaluation_groups.get(group_name)
        with_crop = bool(group_cfg and group_cfg.requires_crops)

        template = load_prompt(
            "classify_one_with_crop" if with_crop else "classify_one"
        )

        # Per-group class context substituted into the shared template.
        class_list = ", ".join(group_cfg.classes) if group_cfg else cand.class_name
        # v4: EvaluationGroupConfig uses .disambiguation instead of .description
        class_descriptions = (group_cfg.disambiguation or "") if group_cfg else ""
        annotation_rules = self._format_rules(group_cfg) if group_cfg else ""

        rendered, prompt_hash = template.render_and_hash(
            class_list=class_list,
            class_descriptions=class_descriptions,
            annotation_rules=annotation_rules,
        )

        # Overview: full image with ONLY this candidate's bbox highlighted.
        overview = draw_focus_on_image(image, cand)
        content = [
            {"type": "image_url",
             "image_url": {"url": pil_to_data_url(overview)}},
        ]
        if with_crop:
            crop = crop_candidate(image, cand.bbox)
            content.append({
                "type": "image_url",
                "image_url": {"url": pil_to_data_url(crop)},
            })
        content.append({
            "type": "text",
            "text": f"Classify the TARGET (proposed class: {cand.class_name}).",
        })

        messages = [
            {"role": "system", "content": rendered},
            {"role": "user", "content": content},
        ]

        vlm_cfg = self.config.servers.vlm
        payload = {
            "model": vlm_cfg.model,
            "messages": messages,
            "temperature": template.model_params.get("temperature", vlm_cfg.temperature),
            "max_tokens": template.model_params.get("max_tokens", 384),
        }
        vlm_url = f"{vlm_cfg.url}/chat/completions"

        vlm_timeout = aiohttp.ClientTimeout(total=120)
        async with sem:
            try:
                async for attempt in http_retry():
                    with attempt:
                        async with asyncio.timeout(120):
                            async with self._session.post(
                                vlm_url, json=payload,
                                headers={"Authorization": f"Bearer {vlm_cfg.api_key}"},
                                timeout=vlm_timeout,
                            ) as resp:
                                resp.raise_for_status()
                                vlm_response = await resp.json()
            except TimeoutError:
                self.logger.warning(
                    "VLM classify_one timed out for %s", cand.candidate_id,
                )
                return None
            except Exception as exc:
                self.logger.warning(
                    "Per-candidate classify failed for %s: %s",
                    cand.candidate_id, exc,
                )
                return None

        try:
            data = parse_vlm_json(vlm_response["choices"][0]["message"]["content"])
            if isinstance(data, list):
                data = data[0] if data else {}
            verdict = VLMVerdict(
                candidate_id=cand.candidate_id,
                correct_class=str(data.get("correct_class", cand.class_name)),
                confidence=float(data.get("confidence", 0.0)),
                bbox_quality=BboxQuality(data.get("bbox_quality", BboxQuality.GOOD)),
                object_complete=bool(data.get("object_complete", True)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as exc:
            self.logger.warning(
                "Malformed verdict for %s: %s", cand.candidate_id, exc,
            )
            return None

        return {
            "group": group_name,
            "prompt_id": template.id,
            "prompt_version": template.version,
            "prompt_hash": prompt_hash,
            "verdict": verdict,
            "tokens": vlm_response.get("usage", {}).get("total_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Verdict resolution
    # ------------------------------------------------------------------

    def _resolve_verdicts(
        self,
        verdicts: list[VLMVerdict],
        candidates: list[Candidate],
    ) -> tuple[list[str], list[str], list[str], dict[str, str]]:
        """Partition verdict IDs into accept / review / reject buckets.

        Pure confidence-driven three-way (section 10.2). ``bbox_quality`` and
        ``object_complete`` remain on the verdict for telemetry but no longer
        drive routing -- spatial decisions belong to the refine stage.

        Returns
        -------
        accepted, review, rejected, relabels
        """
        accepted: list[str] = []
        review: list[str] = []
        rejected: list[str] = []
        relabels: dict[str, str] = {}

        cand_by_id: dict[str, Candidate] = {
            c.candidate_id: c for c in candidates
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
                # Unresolvable -> fall through to confidence routing.

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

    def _format_rules(self, group_cfg: Any) -> str:
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
        # v4: config.classes is dict[str, ClassConfig]
        class_map: dict[str, int] = {
            name: cfg.id for name, cfg in self.config.classes.items()
        }

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
        # human-review queue. Rejected candidates are dropped silently --
        # finalize stage (Phase 3) is the canonical place for drop logging.
        review_candidates = [
            cand
            for cand in detect.candidates
            if cand.candidate_id in evaluate.review
        ]
        if review_candidates:
            self.output_writer.write_review(image_id, review_candidates)
