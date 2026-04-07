from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import adapters as _adapters  # noqa: F401
from . import stages as _stages  # noqa: F401
from .config import AutoAnnotationConfig
from .contracts import FailureRecord, PipelineResult, PipelineState
from .log_utils import get_logger
from .registry import get_adapter, get_stage


logger = get_logger(__name__)


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
        logger.info("run_image.start image_path=%s", image_path)
        image = Image.open(image_path).convert("RGB")
        state = PipelineState(image_path=str(image_path), image=image)

        retry_start_index = next((index for index, stage in enumerate(self.stages) if stage.starts_retry_cycle()), None)
        retry_decider_index = next((index for index, stage in enumerate(self.stages) if stage.decides_retry()), None)

        if retry_start_index is None or retry_decider_index is None or retry_start_index > retry_decider_index:
            for stage in self.stages:
                state, ok = self._run_stage(state, stage)
                if not ok:
                    return self._to_result(state)
            result = self._to_result(state)
            logger.info(
                "run_image.done image_path=%s accepted=%s rejected=%s review=%s",
                image_path,
                len(result.accepted),
                len(result.rejected),
                len(result.human_review),
            )
            return result

        before_retry = self.stages[:retry_start_index]
        retry_cycle = self.stages[retry_start_index : retry_decider_index + 1]
        after_retry = self.stages[retry_decider_index + 1 :]
        retry_decider = retry_cycle[-1]

        for stage in before_retry:
            state, ok = self._run_stage(state, stage)
            if not ok:
                return self._to_result(state)

        state, ok = self._run_retry_cycle(state, retry_cycle, retry_decider)
        if not ok:
            return self._to_result(state)

        for stage in after_retry:
            state, ok = self._run_stage(state, stage)
            if not ok:
                return self._to_result(state)
        result = self._to_result(state)
        logger.info(
            "run_image.done image_path=%s accepted=%s rejected=%s review=%s",
            image_path,
            len(result.accepted),
            len(result.rejected),
            len(result.human_review),
        )
        return result

    def _run_stage(self, state: PipelineState, stage) -> tuple[PipelineState, bool]:
        logger.info("run_image.stage image_path=%s stage=%s", state.image_path, stage.kind)
        try:
            return stage.run(state), True
        except Exception as exc:
            state.failures.append(
                FailureRecord(
                    scope="stage",
                    stage=stage.kind,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    retriable=False,
                )
            )
            state.partial = True
            state.history.append(f"stage_failed:{stage.kind}:{type(exc).__name__}")
            logger.exception("run_image.stage_failed image_path=%s stage=%s", state.image_path, stage.kind)
            return state, False

    def _run_retry_cycle(self, state: PipelineState, retry_cycle: list[object], retry_decider) -> tuple[PipelineState, bool]:
        for stage in retry_cycle:
            state, ok = self._run_stage(state, stage)
            if not ok:
                return state, False

        while retry_decider.requests_retry(state) and state.retry_round < self.config.limits.max_retry_rounds:
            state.retry_round += 1
            state.history.append(f"retry_round={state.retry_round}")
            logger.info("run_image.retry image_path=%s round=%s", state.image_path, state.retry_round)
            for stage in retry_cycle:
                state, ok = self._run_stage(state, stage)
                if not ok:
                    return state, False
        return state, True

    def _to_result(self, state: PipelineState) -> PipelineResult:
        return PipelineResult(
            image_path=state.image_path,
            accepted=state.accepted,
            rejected=state.rejected,
            human_review=state.human_review,
            reviews=state.reviews,
            clusters=state.clusters,
            history=state.history,
            failures=state.failures,
            partial=state.partial,
        )