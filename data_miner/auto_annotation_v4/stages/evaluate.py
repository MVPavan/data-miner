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
    Candidate,
    DetectResult,
    DropReason,
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
        malformed_drops: list[FilterDrop] = []
        for cand, result in zip(to_eval, results):
            if isinstance(result, Exception):
                self.logger.warning(
                    "Per-candidate classify failed for %s: %s",
                    cand.candidate_id, result,
                )
                continue
            if result is None:
                # HTTP failure / timeout. Logged upstream; no candidate drop
                # so a later rerun can retry. (Ghost candidates in the
                # routing summary are surfaced by the viewer's "unbucketed"
                # warning.)
                continue
            if result.get("malformed"):
                # Call succeeded but JSON didn't parse → typed drop.
                vlm_total_tokens += result.get("tokens", 0)
                vlm_calls += 1
                malformed_drops.append(FilterDrop(
                    candidate_id=result["candidate_id"],
                    reason=DropReason.VLM_MALFORMED,
                    context=FilterContext.POST_REVIEW,
                    detail="VLMVerdict parse failed",
                ))
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
        accepted, review, rejected, relabels, verdict_drops = self._resolve_verdicts(
            all_verdicts, candidates_in
        )
        # Malformed parses also count as rejected so downstream stages
        # don't treat them as survivors.
        for d in malformed_drops:
            rejected.append(d.candidate_id)

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

        # Merge verdict-driven rejects (VLM_LOW_CONFIDENCE etc.) and
        # malformed-parse drops into the post-filter drops so the audit
        # trail carries every reason a candidate vanished from survivors.
        drops = list(drops) + verdict_drops + malformed_drops

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

        v2 prompt design: the candidate's **proposed class is hidden** from
        the VLM — we want an independent classification, not a confirmation.
        ``class_match`` is computed in code (see ``_resolve_verdicts``) by
        comparing the VLM's ``detected_class`` against the candidate's
        original class.
        """
        group_name = class_to_group.get(cand.class_name, "default")
        group_cfg = self.config.active_evaluation_groups.get(group_name)
        with_crop = bool(group_cfg and group_cfg.requires_crops)

        template = load_prompt(
            "classify_one_with_crop" if with_crop else "classify_one"
        )

        # Per-group class context substituted into the shared template.
        class_list = ", ".join(group_cfg.classes) if group_cfg else cand.class_name
        class_descriptions = (group_cfg.disambiguation or "") if group_cfg else ""
        per_class_details = self._format_per_class_details(group_cfg)
        annotation_rules = self._format_rules(group_cfg) if group_cfg else ""

        rendered, prompt_hash = template.render_and_hash(
            class_list=class_list,
            class_descriptions=class_descriptions,
            per_class_details=per_class_details,
            annotation_rules=annotation_rules,
        )

        # Image-encoding knobs (JPEG q=90 @ 1280 px by default) from the
        # evaluate config — allows per-job tuning without code edits.
        img_cfg = self.config.evaluate.vlm_image
        img_kw = {
            "max_size": img_cfg.max_size,
            "fmt": img_cfg.format,
            "quality": img_cfg.quality,
        }

        # Overview: full image with ONLY this candidate's bbox highlighted.
        overview = draw_focus_on_image(image, cand)
        content = [
            {"type": "image_url",
             "image_url": {"url": pil_to_data_url(overview, **img_kw)}},
        ]
        if with_crop:
            crop = crop_candidate(image, cand.bbox)
            content.append({
                "type": "image_url",
                "image_url": {"url": pil_to_data_url(crop, **img_kw)},
            })
        # NOTE: proposed class intentionally NOT revealed — classification
        # must be independent of the detector's call to avoid confirmation
        # bias, especially for confusion-pair disambiguation (the whole
        # reason the VLM is in the loop for industrial workloads).
        content.append({
            "type": "text",
            "text": "Classify the object inside the red TARGET box.",
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
            "max_tokens": template.model_params.get("max_tokens", 512),
            # vLLM-compatible structured-output hint. The prompt already
            # asks for a single JSON object; this is the defensive belt
            # that turns most "accidentally wrapped in a fence" failures
            # into clean JSON.
            "response_format": {"type": "json_object"},
        }
        presence_penalty = template.model_params.get("presence_penalty")
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
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
            verdict = VLMVerdict.from_vlm_payload(cand.candidate_id, data)
        except Exception as exc:
            self.logger.warning(
                "Malformed verdict for %s: %s", cand.candidate_id, exc,
            )
            # Signal "call succeeded but JSON parse failed" so the caller
            # can book a VLM_MALFORMED drop (distinct from HTTP failure).
            return {
                "malformed": True,
                "candidate_id": cand.candidate_id,
                "tokens": vlm_response.get("usage", {}).get("total_tokens", 0),
            }

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
    ) -> tuple[list[str], list[str], list[str], dict[str, str], list[FilterDrop]]:
        """Partition verdict IDs into accept / review / reject buckets using
        the v2 two-signal schema: ``class_confidence`` + ``bbox_score``.

        Each reject carries a ``FilterDrop`` with a typed ``DropReason`` so
        the viewer / audit can explain *why* the VLM killed a candidate.

        class_match is computed in code (VLM didn't see the proposed class —
        it classified independently). Class and bbox axes are tuned
        independently so class-correct-but-loose-bbox is a distinct outcome
        from class-wrong.

        Thresholds (all from ``evaluate`` config — independent knobs):
          - ``reject_below``      (0.3): class_confidence floor
          - ``accept_above``      (0.5): class_confidence ceiling
          - ``bbox_reject_below`` (0.3): bbox_score floor
          - ``bbox_accept_above`` (0.5): bbox_score ceiling

        Routing matrix:
          class_match  class_conf      bbox_score        →
          true         ≥ accept        ≥ bbox_accept     accepted
          true         ≥ accept        [bbox_rej,bbacc)  review           (bbox needs work)
          true         ≥ accept        < bbox_reject     rejected (VLM_BBOX_UNUSABLE)
          false        ≥ accept        ≥ bbox_accept     accepted + relabel
          false        ≥ accept        [bbox_rej,bbacc)  review + relabel
          false        ≥ accept        < bbox_reject     rejected (VLM_BBOX_UNUSABLE)
          any          < reject        any               rejected (VLM_LOW_CONFIDENCE)
          any          [reject,accept) any               review
          detected ∈ {other,unknown,none}                rejected (VLM_OTHER_CLASS)
          detected unresolvable via alias_map            rejected (VLM_UNKNOWN_CLASS)
        """
        accepted: list[str] = []
        review: list[str] = []
        rejected: list[str] = []
        relabels: dict[str, str] = {}
        drops: list[FilterDrop] = []

        cand_by_id: dict[str, Candidate] = {
            c.candidate_id: c for c in candidates
        }
        eval_cfg = self.config.evaluate
        accept_thr = eval_cfg.accept_above
        reject_thr = eval_cfg.reject_below
        bbox_accept = eval_cfg.bbox_accept_above
        bbox_reject = eval_cfg.bbox_reject_below

        def _reject(cid: str, reason: DropReason, note: str) -> None:
            rejected.append(cid)
            drops.append(FilterDrop(
                candidate_id=cid,
                reason=reason,
                context=FilterContext.POST_REVIEW,
                detail=note,
            ))

        for v in verdicts:
            cand = cand_by_id.get(v.candidate_id)
            original_class = cand.class_name if cand else ""
            detected = (v.detected_class or "").strip()
            class_conf = float(v.class_confidence or 0.0)
            # A VLM that OMITS bbox_score must not silently pass the bbox
            # gate — default to 0.0 so a missing field reads as "unusable"
            # rather than "perfect". The VLMVerdict default of 1.0 covers
            # the "field present but coerced" case inside the model; this
            # branch covers the rarer `None` that survives migration.
            bbox_score = float(v.bbox_score if v.bbox_score is not None else 0.0)

            # ---- 1. Hard class-level rejections ----
            if not detected or detected.lower() in ("other", "unknown", "none"):
                _reject(
                    v.candidate_id, DropReason.VLM_OTHER_CLASS,
                    f"detected_class={detected!r}",
                )
                continue
            canonical = resolve_canonical_class(detected, self.alias_map)
            if canonical is None:
                _reject(
                    v.candidate_id, DropReason.VLM_UNKNOWN_CLASS,
                    f"detected_class={detected!r} not in alias map",
                )
                continue

            class_match = (canonical == original_class)

            # ---- 2. Low class-confidence overrides everything else ----
            if class_conf < reject_thr:
                _reject(
                    v.candidate_id, DropReason.VLM_LOW_CONFIDENCE,
                    f"class_confidence={class_conf:.2f} < {reject_thr:.2f}",
                )
                continue

            # ---- 3. Medium class-confidence → review (with relabel hint) ----
            if class_conf < accept_thr:
                if not class_match:
                    relabels[v.candidate_id] = canonical
                review.append(v.candidate_id)
                continue

            # ---- 4. High class-confidence: now gate on bbox quality ----
            if bbox_score < bbox_reject:
                _reject(
                    v.candidate_id, DropReason.VLM_BBOX_UNUSABLE,
                    f"bbox_score={bbox_score:.2f} < {bbox_reject:.2f}",
                )
                continue

            if not class_match:
                relabels[v.candidate_id] = canonical

            if bbox_score >= bbox_accept:
                accepted.append(v.candidate_id)
            else:
                # Good class call, loose/tight bbox → human review (or the
                # refine stage picks it up for refine-eligible classes).
                review.append(v.candidate_id)

        return accepted, review, rejected, relabels, drops

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

    def _format_per_class_details(self, group_cfg: Any) -> str:
        """Build the PER-CLASS DETAILS block from class_registry.description.

        For each class in ``group_cfg.classes``, look up its description in
        ``config.class_registry`` and emit a bullet. Classes without a
        description are omitted (common objects like ``car`` / ``bicycle``
        the VLM already knows don't need extra cues).

        Returns an empty string if the group or registry is missing, or no
        class in the group has a description.
        """
        if not group_cfg:
            return ""
        registry = getattr(self.config, "class_registry", None) or {}
        lines: list[str] = []
        for cls_name in group_cfg.classes:
            cls_cfg = registry.get(cls_name)
            desc = (getattr(cls_cfg, "description", "") or "").strip()
            if desc:
                lines.append(f"- {cls_name}: {desc}")
        return "\n".join(lines)

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
