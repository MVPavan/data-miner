from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import adapters as _adapters  # noqa: F401
from . import stages as _stages  # noqa: F401
from .config import AutoAnnotationConfig
from .contracts import PipelineResult, PipelineState
from .registry import get_adapter, get_stage


class AutoAnnotationPipeline:
    def __init__(self, config: AutoAnnotationConfig):
        self.config = config
        self.adapters = self._build_adapters()
        self.stages = self._build_stages()

    def _build_adapters(self) -> dict[str, object]:
        adapters: dict[str, object] = {}
        for name, model_cfg in self.config.models.items():
            if not model_cfg.enabled:
                continue
            adapter_cls = get_adapter(model_cfg.kind)
            adapters[name] = adapter_cls(name=name, config=model_cfg)
        return adapters

    def _build_stages(self) -> list[object]:
        stages = []
        for stage_cfg in self.config.stages:
            if not stage_cfg.enabled:
                continue
            stage_cls = get_stage(stage_cfg.implementation or stage_cfg.kind)
            stages.append(stage_cls(config=stage_cfg, adapters=self.adapters, pipeline_config=self.config))
        return stages

    def run_image(self, image_path: str | Path) -> PipelineResult:
        image = Image.open(image_path).convert("RGB")
        state = PipelineState(image_path=str(image_path), image=image)

        verification_index = next((index for index, stage in enumerate(self.stages) if stage.kind == "verification"), None)
        if verification_index is None:
            for stage in self.stages:
                state = stage.run(state)
            return self._to_result(state)

        pre_verify = self.stages[:verification_index]
        verifier = self.stages[verification_index]
        post_verify = self.stages[verification_index + 1 :]

        for stage in pre_verify:
            state = stage.run(state)
        state = verifier.run(state)

        refinement_stage = next((stage for stage in pre_verify if stage.kind == "refinement"), None)
        while refinement_stage and self._has_retry_requests(state) and state.retry_round < self.config.limits.max_retry_rounds:
            state.retry_round += 1
            state.history.append(f"retry_round={state.retry_round}")
            state = refinement_stage.run(state)
            state = verifier.run(state)

        for stage in post_verify:
            state = stage.run(state)
        return self._to_result(state)

    def _has_retry_requests(self, state: PipelineState) -> bool:
        return any(review.recommended_action == "refine" for review in state.reviews.values())

    def _to_result(self, state: PipelineState) -> PipelineResult:
        return PipelineResult(
            image_path=state.image_path,
            accepted=state.accepted,
            rejected=state.rejected,
            human_review=state.human_review,
            reviews=state.reviews,
            clusters=state.clusters,
            history=state.history,
        )