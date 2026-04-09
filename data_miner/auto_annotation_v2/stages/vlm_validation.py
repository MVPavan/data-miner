"""VLM validation stage: re-evaluate refined candidates using the same two-pass approach."""

from __future__ import annotations

from PIL import Image

from ..config import AutoAnnotationV2Config
from ..contracts import (
    Candidate,
    CandidateStatus,
    DetailedVerdict,
    ScreeningVerdict,
)
from ..log_utils import get_logger
from .vlm_reasoning import run_vlm_reasoning

logger = get_logger(__name__)


async def run_vlm_validation(
    image: Image.Image,
    candidates: list[Candidate],
    config: AutoAnnotationV2Config,
) -> tuple[list[ScreeningVerdict], list[DetailedVerdict]]:
    """Validate refined candidates using the same VLM reasoning pipeline.

    Only validates candidates that were refined (status == REFINED).
    Non-refined candidates are not re-evaluated.

    Returns:
        (validation_screening, validation_detailed)
    """
    refined = [c for c in candidates if c.status == CandidateStatus.REFINED]

    if not refined:
        logger.info("No refined candidates to validate")
        return [], []

    logger.info("Validating %d refined candidates", len(refined))
    screening, detailed = await run_vlm_reasoning(image, refined, config)
    logger.info(
        "Validation complete: %d screening verdicts, %d detailed verdicts",
        len(screening),
        len(detailed),
    )
    return screening, detailed
