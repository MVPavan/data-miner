"""Pipeline runner with per-stage-per-image checkpoint resume."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .checkpoint import CheckpointManager
from .config import AutoAnnotationV2Config
from .contracts import (
    Candidate,
    DetailedVerdict,
    FailureRecord,
    FinalAnnotation,
    ImageTrace,
    PipelineResult,
    RefinementAction,
    ScreeningVerdict,
    StageName,
    StageRecord,
)
from .log_utils import get_logger
from .stages.filtering import run_filtering
from .stages.finalize import build_final_annotations, build_pipeline_result, save_result
from .stages.proposal import run_proposal
from .stages.vlm_reasoning import run_vlm_reasoning
from .stages.vlm_refinement import run_vlm_refinement
from .stages.vlm_validation import run_vlm_validation

logger = get_logger(__name__)

STAGE_ORDER: list[StageName] = [
    StageName.PROPOSAL,
    StageName.FILTERING,
    StageName.VLM_REASONING,
    StageName.VLM_REFINEMENT,
    StageName.VLM_VALIDATION,
    StageName.FINALIZE,
]


class _StageContext:
    """Helper to reduce boilerplate per stage."""

    def __init__(
        self,
        stem: str,
        stage: StageName,
        trace: ImageTrace,
        checkpoint: CheckpointManager,
        force_redo: bool = False,
    ) -> None:
        self.stem = stem
        self.stage = stage
        self.trace = trace
        self.checkpoint = checkpoint
        self.force_redo = force_redo
        self.started: str = ""
        self.failed = False

    def has_checkpoint(self) -> bool:
        if self.force_redo:
            logger.info(
                "[%s] Force redo: %s (clearing checkpoint + downstream)",
                self.stem,
                self.stage.value,
            )
            self.checkpoint.clear_stage_and_downstream(
                self.stem, self.stage, STAGE_ORDER
            )
            return False
        if self.checkpoint.exists(self.stem, self.stage):
            logger.info(
                "[%s] Resuming from checkpoint: %s", self.stem, self.stage.value
            )
            return True
        return False

    def begin(self) -> None:
        self.started = datetime.now(timezone.utc).isoformat()

    def save(self, data: Any) -> None:
        self.checkpoint.save(self.stem, self.stage, data)

    def record_success(self, count_in: int, count_out: int) -> None:
        self.trace.stages.append(
            StageRecord(
                stage=self.stage,
                started_at=self.started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                candidate_count_in=count_in,
                candidate_count_out=count_out,
            )
        )

    def record_failure(self, exc: Exception) -> None:
        logger.exception("[%s] %s failed", self.stem, self.stage.value)
        self.trace.failures.append(
            FailureRecord(
                stage=self.stage,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        )
        self.failed = True


class AutoAnnotationPipelineV2:
    """Main pipeline runner with checkpoint-based crash recovery."""

    def __init__(
        self,
        config: AutoAnnotationV2Config,
        output_dir: Path,
        force_redo_stages: set[StageName] | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.checkpoint = CheckpointManager(output_dir / ".checkpoints")
        self._force_redo = force_redo_stages or set()

    def _is_stage_enabled(self, stage: StageName) -> bool:
        return getattr(self.config.stages, stage.value, True)

    def _enabled_stages(self) -> list[StageName]:
        return [s for s in STAGE_ORDER if self._is_stage_enabled(s)]

    def _ctx(self, stem: str, stage: StageName, trace: ImageTrace) -> _StageContext:
        return _StageContext(
            stem,
            stage,
            trace,
            self.checkpoint,
            force_redo=stage in self._force_redo,
        )

    async def run_image(self, image_path: str | Path) -> PipelineResult:
        """Run the full pipeline for one image with checkpoint resume."""
        image_path = Path(image_path)
        stem = image_path.stem
        logger.info("Processing image: %s", image_path)

        image = Image.open(image_path).convert("RGB")
        trace = ImageTrace(image_path=str(image_path))
        partial = False

        # State flowing between stages
        proposal_candidates: list[Candidate] = []
        filtered_candidates: list[Candidate] = []
        screening_verdicts: list[ScreeningVerdict] = []
        detailed_verdicts: list[DetailedVerdict] = []
        refined_candidates: list[Candidate] = []
        refinement_actions: list[RefinementAction] = []
        validation_screening: list[ScreeningVerdict] = []
        validation_detailed: list[DetailedVerdict] = []
        final_annotations: list[FinalAnnotation] = []

        # ── PROPOSAL ─────────────────────────────────────────────
        if self._is_stage_enabled(StageName.PROPOSAL):
            ctx = self._ctx(stem, StageName.PROPOSAL, trace)

            if ctx.has_checkpoint():
                proposal_candidates = self.checkpoint.load_list_as(
                    stem, StageName.PROPOSAL, Candidate
                )
            else:
                ctx.begin()
                try:
                    proposal_candidates = run_proposal(image, self.config)
                    ctx.save(proposal_candidates)
                    ctx.record_success(count_in=0, count_out=len(proposal_candidates))
                except Exception as exc:
                    ctx.record_failure(exc)
                    partial = True

            trace.proposal_candidates = proposal_candidates

        # ── FILTERING ────────────────────────────────────────────
        if self._is_stage_enabled(StageName.FILTERING):
            ctx = self._ctx(stem, StageName.FILTERING, trace)

            if ctx.has_checkpoint():
                filtered_candidates = self.checkpoint.load_list_as(
                    stem, StageName.FILTERING, Candidate
                )
            else:
                ctx.begin()
                try:
                    filtered_candidates = run_filtering(
                        proposal_candidates, self.config
                    )
                    ctx.save(filtered_candidates)
                    ctx.record_success(
                        count_in=len(proposal_candidates),
                        count_out=len(filtered_candidates),
                    )
                except Exception as exc:
                    ctx.record_failure(exc)
                    filtered_candidates = proposal_candidates  # fallback
                    partial = True

            trace.filtered_candidates = filtered_candidates

        # ── VLM REASONING ────────────────────────────────────────
        if self._is_stage_enabled(StageName.VLM_REASONING):
            ctx = self._ctx(stem, StageName.VLM_REASONING, trace)

            if ctx.has_checkpoint():
                data = self.checkpoint.load(stem, StageName.VLM_REASONING)
                screening_verdicts = [
                    ScreeningVerdict.model_validate(v)
                    for v in data.get("screening", [])
                ]
                detailed_verdicts = [
                    DetailedVerdict.model_validate(v) for v in data.get("detailed", [])
                ]
            else:
                ctx.begin()
                try:
                    screening_verdicts, detailed_verdicts = await run_vlm_reasoning(
                        image, filtered_candidates, self.config
                    )
                    ctx.save(
                        {
                            "screening": [
                                v.model_dump(mode="json") for v in screening_verdicts
                            ],
                            "detailed": [
                                v.model_dump(mode="json") for v in detailed_verdicts
                            ],
                        }
                    )
                    ctx.record_success(
                        count_in=len(filtered_candidates),
                        count_out=len(screening_verdicts),
                    )
                except Exception as exc:
                    ctx.record_failure(exc)
                    partial = True

            trace.screening_results = screening_verdicts
            trace.detailed_verdicts = detailed_verdicts

        # ── VLM REFINEMENT ───────────────────────────────────────
        if self._is_stage_enabled(StageName.VLM_REFINEMENT):
            ctx = self._ctx(stem, StageName.VLM_REFINEMENT, trace)

            if ctx.has_checkpoint():
                data = self.checkpoint.load(stem, StageName.VLM_REFINEMENT)
                refined_candidates = [
                    Candidate.model_validate(c) for c in data.get("candidates", [])
                ]
                refinement_actions = [
                    RefinementAction.model_validate(a) for a in data.get("actions", [])
                ]
            else:
                ctx.begin()
                try:
                    refined_candidates, refinement_actions = await run_vlm_refinement(
                        image,
                        filtered_candidates,
                        screening_verdicts,
                        detailed_verdicts,
                        self.config,
                    )
                    ctx.save(
                        {
                            "candidates": [
                                c.model_dump(mode="json") for c in refined_candidates
                            ],
                            "actions": [
                                a.model_dump(mode="json") for a in refinement_actions
                            ],
                        }
                    )
                    ctx.record_success(
                        count_in=len(filtered_candidates),
                        count_out=len(refined_candidates),
                    )
                except Exception as exc:
                    ctx.record_failure(exc)
                    refined_candidates = filtered_candidates  # fallback
                    partial = True

            trace.refined_candidates = refined_candidates
            trace.refinement_proposals = refinement_actions

        # ── VLM VALIDATION ───────────────────────────────────────
        if self._is_stage_enabled(StageName.VLM_VALIDATION):
            ctx = self._ctx(stem, StageName.VLM_VALIDATION, trace)

            if ctx.has_checkpoint():
                data = self.checkpoint.load(stem, StageName.VLM_VALIDATION)
                validation_screening = [
                    ScreeningVerdict.model_validate(v)
                    for v in data.get("screening", [])
                ]
                validation_detailed = [
                    DetailedVerdict.model_validate(v) for v in data.get("detailed", [])
                ]
            else:
                ctx.begin()
                try:
                    (
                        validation_screening,
                        validation_detailed,
                    ) = await run_vlm_validation(image, refined_candidates, self.config)
                    ctx.save(
                        {
                            "screening": [
                                v.model_dump(mode="json") for v in validation_screening
                            ],
                            "detailed": [
                                v.model_dump(mode="json") for v in validation_detailed
                            ],
                        }
                    )
                    ctx.record_success(
                        count_in=len(refined_candidates),
                        count_out=len(validation_screening),
                    )
                except Exception as exc:
                    ctx.record_failure(exc)
                    partial = True

            trace.validation_verdicts = validation_detailed

        # ── FINALIZE ─────────────────────────────────────────────
        final_candidates = (
            refined_candidates or filtered_candidates or proposal_candidates
        )

        if self._is_stage_enabled(StageName.FINALIZE):
            started = datetime.now(timezone.utc).isoformat()
            final_annotations = build_final_annotations(
                final_candidates,
                screening_verdicts,
                detailed_verdicts,
                validation_screening,
                validation_detailed,
            )
            trace.final_annotations = final_annotations
            trace.stages.append(
                StageRecord(
                    stage=StageName.FINALIZE,
                    started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    candidate_count_in=len(final_candidates),
                    candidate_count_out=len(final_annotations),
                )
            )

        result = build_pipeline_result(str(image_path), trace, partial=partial)

        # Save outputs
        class_names = [cp.name for cp in self.config.classes]
        save_result(result, class_names, self.output_dir, self.config.output)

        logger.info(
            "[%s] Complete: %d accepted, %d rejected, %d review, partial=%s",
            stem,
            len(result.accepted),
            len(result.rejected),
            len(result.human_review),
            partial,
        )
        return result

    async def run_batch(self, image_paths: list[Path]) -> list[PipelineResult]:
        """Run the pipeline for multiple images sequentially."""
        results: list[PipelineResult] = []
        for idx, image_path in enumerate(image_paths, start=1):
            logger.info("Image %d/%d: %s", idx, len(image_paths), image_path.name)
            try:
                result = await self.run_image(image_path)
                results.append(result)
            except Exception:
                logger.exception("Failed to process %s", image_path)
        return results

    def run_image_sync(self, image_path: str | Path) -> PipelineResult:
        """Synchronous wrapper for run_image."""
        return asyncio.run(self.run_image(image_path))

    def run_batch_sync(self, image_paths: list[Path]) -> list[PipelineResult]:
        """Synchronous wrapper for run_batch."""
        return asyncio.run(self.run_batch(image_paths))
