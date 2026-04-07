from __future__ import annotations

from typing import Any

from PIL import Image

from ..config import ClassPackConfig, ModelConfig
from ..contracts import Candidate, ReviewDecision


class AnnotationAdapter:
    capabilities: set[str] = set()

    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def propose(
        self,
        image: Image.Image,
        class_pack: ClassPackConfig,
        expression: str,
        params: dict[str, Any],
    ) -> list[Candidate]:
        raise NotImplementedError

    def refine(
        self,
        image: Image.Image,
        candidate: Candidate,
        class_pack: ClassPackConfig,
        params: dict[str, Any],
        request: ReviewDecision | None = None,
    ) -> Candidate | None:
        raise NotImplementedError

    def verify(
        self,
        image: Image.Image,
        candidate: Candidate,
        class_pack: ClassPackConfig,
        params: dict[str, Any],
    ) -> ReviewDecision:
        raise NotImplementedError