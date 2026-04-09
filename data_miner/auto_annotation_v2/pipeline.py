"""Pipeline runner with per-stage-per-image checkpoint resume."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

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
from .log_utils import configure_logging, get_logger
from .stages.filtering import run_filtering
from .stages.finalize import build_final_annotations, build_pipeline_result, save_result
from .stages.proposal import run_proposal
from .stages.vlm_reasoning import run_vlm_reasoning
from .stages.vlm_refinement import run_vlm_refinement
from .stages.vlm_validation import run_vlm_validation

logger = get_logger(__name__)

# Stage execution order
STAGE_ORDER: list[StageName] = [
    StageName.PROPOSAL,
    StageName.FILTERING,
    StageName.VLM_REASONING,
    StageName.VLM_REFINEMENT,
    StageName.VLM_VALIDATION,
    StageName.FINALIZE,
]


class AutoAnnotationPipelineV2:
    """Main pipeline runner with checkpoint-based crash recovery."""

    def __init__(
        self,
        config: AutoAnnotationV2Config,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.checkpoint = CheckpointManager(output_dir / ".checkpoints")

    def _is_stage_enabled(self, stage: StageName) -> bool:
        return getattr(self.config.stages, stage.value, True)

    def _enabled_stages(self) -> list[StageName]:
        return [s for s in STAGE_ORDER if self._is_stage_enabled(s)]

    async def run_image(self, image_path: str | Path) -> PipelineResult:
        """Run the full pipeline for one image with checkpoint resume."""
        image_path = Path(image_path)
        stem = image_path.stem
        logger.info("Processing image: %s", image_path)

        image = Image.open(image_path).convert("RGB")
        trace = ImageTrace(image_path=str(image_path))
        partial = False

        # State that flows between stages
        proposal_candidates: list[Candidate] = []
        filtered_candidates: list[Candidate] = []
        screening_verdicts: list[ScreeningVerdict] = []
        detailed_verdicts: list[DetailedVerdict] = []
        refined_candidates: list[Candidate] = []
        refinement_actions: list[RefinementAction] = []
        validation_screening: list[ScreeningVerdict] = []
        validation_detailed: list[DetailedVerdict] = []
        final_annotations: list[FinalAnnotation] = []

        # --- PROPOSAL ---
        if self._is_stage_enabled(StageName.PROPOSAL):
            if self.checkpoint.exists(stem, StageName.PROPOSAL):
                logger.info("[%s] Resuming from checkpoint: proposal", stem)
                proposal_candidates = self.checkpoint.load_list_as(
                    stem, StageName.PROPOSAL, Candidate
                )
            else:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    proposal_candidates = run_proposal(image, self.config)
                    self.checkpoint.save(stem, StageName.PROPOSAL, proposal_candidates)
                except Exception as exc:
                    logger.exception("[%s] Proposal failed", stem)
                    trace.failures.append(FailureRecord(
                        stage=StageName.PROPOSAL, error_type=type(exc).__name__,
                        message=str(exc),
                    ))
                    partial = True
                else:
                    trace.stages.append(StageRecord(
                        stage=StageName.PROPOSAL, started_at=started,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        candidate_count_in=0, candidate_count_out=len(proposal_candidates),
                    ))
            trace.proposal_candidates = proposal_candidates

        # --- FILTERING ---
        if self._is_stage_enabled(StageName.FILTERING):
            if self.checkpoint.exists(stem, StageName.FILTERING):
                logger.info("[%s] Resuming from checkpoint: filtering", stem)
                filtered_candidates = self.checkpoint.load_list_as(
                    stem, StageName.FILTERING, Candidate
                )
            else:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    filtered_candidates = run_filtering(proposal_candidates, self.config)
                    self.checkpoint.save(stem, StageName.FILTERING, filtered_candidates)
                except Exception as exc:
                    logger.exception("[%s] Filtering failed", stem)
                    trace.failures.append(FailureRecord(
                        stage=StageName.FILTERING, error_type=type(exc).__name__,
                        message=str(exc),
                    ))
                    filtered_candidates = proposal_candidates
                    partial = True
                else:
                    trace.stages.append(StageRecord(
                        stage=StageName.FILTERING, started_at=started,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        candidate_count_in=len(proposal_candidates),
                        candidate_count_out=len(filtered_candidates),
                    ))
            trace.filtered_candidates = filtered_candidates

        # --- VLM REASONING ---
        if self._is_stage_enabled(StageName.VLM_REASONING):
            if self.checkpoint.exists(stem, StageName.VLM_REASONING):
                logger.info("[%s] Resuming from checkpoint: vlm_reasoning", stem)
                data = self.checkpoint.load(stem, StageName.VLM_REASONING)
                screening_verdicts = [
                    ScreeningVerdict.model_validate(v) for v in data.get("screening", [])
                ]
                detailed_verdicts = [
                    DetailedVerdict.model_validate(v) for v in data.get("detailed", [])
                ]
            else:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    screening_verdicts, detailed_verdicts = await run_vlm_reasoning(
                        image, filtered_candidates, self.config
                    )
                    self.checkpoint.save(stem, StageName.VLM_REASONING, {
                        "screening": [v.model_dump(mode="json") for v in screening_verdicts],
                        "detailed": [v.model_dump(mode="json") for v in detailed_verdicts],
                    })
                except Exception as exc:
                    logger.exception("[%s] VLM reasoning failed", stem)
                    trace.failures.append(FailureRecord(
                        stage=StageName.VLM_REASONING, error_type=type(exc).__name__,
                        message=str(exc),
                    ))
                    partial = True
                else:
                    trace.stages.append(StageRecord(
                        stage=StageName.VLM_REASONING, started_at=started,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        candidate_count_in=len(filtered_candidates),
                        candidate_count_out=len(screening_verdicts),
                    ))
            trace.screening_results = screening_verdicts
            trace.detailed_verdicts = detailed_verdicts

        # --- VLM REFINEMENT ---
        if self._is_stage_enabled(StageName.VLM_REFINEMENT):
            if self.checkpoint.exists(stem, StageName.VLM_REFINEMENT):
                logger.info("[%s] Resuming from checkpoint: vlm_refinement", stem)
                data = self.checkpoint.load(stem, StageName.VLM_REFINEMENT)
                refined_candidates = [
                    Candidate.model_validate(c) for c in data.get("candidates", [])
                ]
                refinement_actions = [
                    RefinementAction.model_validate(a) for a in data.get("actions", [])
                ]
            else:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    refined_candidates, refinement_actions = await run_vlm_refinement(
                        image, filtered_candidates, screening_verdicts,
                        detailed_verdicts, self.config,
                    )
                    self.checkpoint.save(stem, StageName.VLM_REFINEMENT, {
                        "candidates": [c.model_dump(mode="json") for c in refined_candidates],
                        "actions": [a.model_dump(mode="json") for a in refinement_actions],
                    })
                except Exception as exc:
                    logger.exception("[%s] VLM refinement failed", stem)
                    trace.failures.append(FailureRecord(
                        stage=StageName.VLM_REFINEMENT, error_type=type(exc).__name__,
                        message=str(exc),
                    ))
                    refined_candidates = filtered_candidates
                    partial = True
                else:
                    trace.stages.append(StageRecord(
                        stage=StageName.VLM_REFINEMENT, started_at=started,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        candidate_count_in=len(filtered_candidates),
                        candidate_count_out=len(refined_candidates),
                    ))
            trace.refined_candidates = refined_candidates
            trace.refinement_proposals = refinement_actions

        # --- VLM VALIDATION ---
        if self._is_stage_enabled(StageName.VLM_VALIDATION):
            if self.checkpoint.exists(stem, StageName.VLM_VALIDATION):
                logger.info("[%s] Resuming from checkpoint: vlm_validation", stem)
                data = self.checkpoint.load(stem, StageName.VLM_VALIDATION)
                validation_screening = [
                    ScreeningVerdict.model_validate(v) for v in data.get("screening", [])
                ]
                validation_detailed = [
                    DetailedVerdict.model_validate(v) for v in data.get("detailed", [])
                ]
            else:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    validation_screening, validation_detailed = await run_vlm_validation(
                        image, refined_candidates, self.config
                    )
                    self.checkpoint.save(stem, StageName.VLM_VALIDATION, {
                        "screening": [v.model_dump(mode="json") for v in validation_screening],
                        "detailed": [v.model_dump(mode="json") for v in validation_detailed],
                    })
                except Exception as exc:
                    logger.exception("[%s] VLM validation failed", stem)
                    trace.failures.append(FailureRecord(
                        stage=StageName.VLM_VALIDATION, error_type=type(exc).__name__,
                        message=str(exc),
                    ))
                    partial = True
                else:
                    trace.stages.append(StageRecord(
                        stage=StageName.VLM_VALIDATION, started_at=started,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        candidate_count_in=len(refined_candidates),
                        candidate_count_out=len(validation_screening),
                    ))
            trace.validation_verdicts = validation_detailed

        # --- FINALIZE ---
        final_candidates = refined_candidates or filtered_candidates or proposal_candidates

        if self._is_stage_enabled(StageName.FINALIZE):
            started = datetime.now(timezone.utc).isoformat()
            final_annotations = build_final_annotations(
                final_candidates,
                screening_verdicts, detailed_verdicts,
                validation_screening, validation_detailed,
            )
            trace.final_annotations = final_annotations
            trace.stages.append(StageRecord(
                stage=StageName.FINALIZE, started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                candidate_count_in=len(final_candidates),
                candidate_count_out=len(final_annotations),
            ))

        result = build_pipeline_result(str(image_path), trace, partial=partial)

        # Save outputs
        class_names = [cp.name for cp in self.config.classes]
        save_result(result, class_names, self.output_dir, self.config.output)

        logger.info(
            "[%s] Complete: %d accepted, %d rejected, %d review, partial=%s",
            stem, len(result.accepted), len(result.rejected),
            len(result.human_review), partial,
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
