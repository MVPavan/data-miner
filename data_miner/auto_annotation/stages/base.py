from __future__ import annotations

from ..config import AutoAnnotationConfig, StageConfig
from ..contracts import PipelineState


class Stage:
    kind = "base"

    def __init__(self, config: StageConfig, adapters: dict[str, object], pipeline_config: AutoAnnotationConfig):
        self.config = config
        self.adapters = adapters
        self.pipeline_config = pipeline_config

    def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError