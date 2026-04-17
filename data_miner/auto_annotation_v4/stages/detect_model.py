"""Stage 1a: Per-model detection -- call ONE model server, save proposal, check barrier.

Phase 2 splits the monolithic detect stage into per-model workers
(DetectModelWorker) and a merge worker (DetectMergeWorker in detect.py).

Each DetectModelWorker:
  1. Claims from ``work_queue`` where ``stage = "detect:{model_name}"``
  2. Calls its model server's ``/predict`` endpoint
  3. Saves the raw proposal to the ``proposals`` table
  4. Checks the barrier: have all enabled models submitted proposals?
  5. If yes -> queues ``"detect:merge"`` for this image
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp
from pydantic import BaseModel

from ..configs import (
    AutoAnnotationV4Config,
    BoundingBox,
    Candidate,
    CandidateStatus,
    ClassConfig,
    DetectorName,
    DetectorRequest,
    DetectorResponse,
    ProposalResult,
    Stage,
    StageMessage,
)
from ..utils import get_image_size
from ..workers.base import StageWorker

logger = logging.getLogger("data_miner.auto_annotation_v4.detect_model")


class DetectModelWorker(StageWorker):
    """Per-model detect worker. Calls ONE model server and saves proposal.

    Unlike normal StageWorker subclasses, this worker:
    - Saves to the ``proposals`` table (not ``stages``)
    - Uses a custom :meth:`run` loop instead of the base save_and_forward flow
    - Checks a barrier after saving and conditionally queues the merge step

    Parameters
    ----------
    config:
        Full pipeline configuration.
    db:
        Connected CheckpointDB instance.
    model_name:
        Which detector this worker handles (e.g. DetectorName.GROUNDING_DINO).
    enabled_models:
        All detectors that must complete before the merge barrier opens.
    """

    def __init__(
        self,
        config: AutoAnnotationV4Config,
        db: Any,
        *,
        model_name: DetectorName,
        enabled_models: list[DetectorName],
        server_semaphore: asyncio.Semaphore | None = None,
        worker_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._enabled_models = enabled_models
        # Set instance-level stage BEFORE super().__init__ validates it.
        self.stage = f"detect:{model_name.value}"
        super().__init__(config, db, server_semaphore=server_semaphore, worker_id=worker_id, job_id=job_id)

        # Build detector config ref
        self._detector_cfg = config.servers.detectors.get(model_name)

    # ------------------------------------------------------------------
    # Custom run loop (overrides StageWorker.run)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: claim -> check proposal cache -> call server -> save proposal -> barrier.

        Overrides the base StageWorker.run() because:
        - We save to ``proposals`` not ``stages``
        - We don't use ``save_and_forward()``
        - Barrier check + merge queueing replaces next-stage routing
        """
        self._running = True
        self.logger.info(
            "Worker %s started on stage '%s' (model=%s)",
            self.worker_id,
            self.stage,
            self.model_name.value,
        )

        self._session = aiohttp.ClientSession()
        try:
            while self._running:
                image_id = await self.db.claim_work(self.stage, self.worker_id)
                if image_id is None:
                    await asyncio.sleep(1)
                    continue

                try:
                    # Skip if proposal already cached
                    if await self.db.proposal_exists(image_id, self.model_name):
                        self.logger.debug(
                            "Proposal %s/%s already cached -- checking barrier",
                            image_id,
                            self.model_name.value,
                        )
                        await self.db.complete_work(image_id, self.stage)
                        await self._check_and_queue_merge(image_id)
                        continue

                    # Resolve image path
                    image_path = await self.db.resolve_image_path(image_id)

                    # Call model server
                    t0 = time.perf_counter()
                    candidates = await self._call_model(image_id, image_path)
                    latency_ms = (time.perf_counter() - t0) * 1000

                    # Save proposal
                    image_size = list(get_image_size(image_path))
                    await self.db.save_proposal(
                        image_id,
                        self.model_name,
                        ProposalResult(
                            model=self.model_name.value,
                            image_id=image_id,
                            image_size=image_size,
                            latency_ms=latency_ms,
                            candidates=candidates,
                        ),
                    )
                    self.logger.info(
                        "Model %s returned %d candidates for %s in %.0f ms",
                        self.model_name.value,
                        len(candidates),
                        image_id,
                        latency_ms,
                    )

                    # Mark work done and check barrier
                    await self.db.complete_work(image_id, self.stage)
                    await self._check_and_queue_merge(image_id)

                except asyncio.CancelledError:
                    await self.db.release_work(image_id, self.stage)
                    raise

                except Exception as exc:
                    self.logger.exception(
                        "Failed %s/%s", image_id, self.stage
                    )
                    await self.db.fail_work(image_id, self.stage, str(exc))

        except asyncio.CancelledError:
            self.logger.info("%s shutting down", self.worker_id)
        finally:
            await self._session.close()
            self._session = None

        self.logger.info("Worker %s stopped", self.worker_id)

    # ------------------------------------------------------------------
    # Barrier check
    # ------------------------------------------------------------------

    async def _check_and_queue_merge(self, image_id: str) -> None:
        """If all enabled models have proposals, queue the merge step."""
        model_values = [m.value for m in self._enabled_models]
        if await self.db.barrier_ready(image_id, model_values):
            await self.db.add_work("detect:merge", image_id)
            self.logger.debug(
                "Barrier met for %s -- queued detect:merge", image_id
            )

    # ------------------------------------------------------------------
    # Model call
    # ------------------------------------------------------------------

    async def _call_model(
        self, image_id: str, image_path: str
    ) -> list[Candidate]:
        """Call this worker's model server and return candidates."""
        if self._detector_cfg is None:
            self.logger.warning(
                "No config for detector %s -- returning empty",
                self.model_name.value,
            )
            return []

        active_classes = self.config.classes
        if not active_classes:
            return []

        # Build flat prompt list from all active classes
        all_prompts: list[str] = []
        for cls_cfg in active_classes.values():
            all_prompts.extend(cls_cfg.prompts)

        req = DetectorRequest(
            image_path=image_path,
            prompts=all_prompts,
        )

        url = f"http://localhost:{self._detector_cfg.port}/predict"
        request_timeout = aiohttp.ClientTimeout(total=60)

        acquired = False
        try:
            if self._server_semaphore is not None:
                await self._server_semaphore.acquire()
                acquired = True
            async with asyncio.timeout(60):
                async with self._session.post(
                    url, json=req.model_dump(), timeout=request_timeout,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except TimeoutError:
            self.logger.warning(
                "%s timed out for %s", self.model_name.value, image_id,
            )
            return []
        except Exception as exc:
            self.logger.warning(
                "%s POST failed for %s: %s",
                self.model_name.value,
                image_id,
                exc,
            )
            return []
        finally:
            if acquired:
                self._server_semaphore.release()

        return _to_candidates(
            DetectorResponse.model_validate(data),
            self.model_name,
            active_classes,
        )

    # ------------------------------------------------------------------
    # Not used (custom run loop) but required by ABC
    # ------------------------------------------------------------------

    async def process(self, msg: StageMessage) -> BaseModel:
        """Not used -- DetectModelWorker overrides run() directly."""
        raise NotImplementedError("DetectModelWorker uses custom run()")

    def _resolve_next_stage(self, result: BaseModel) -> Stage | str:
        """Not used -- barrier check handles routing."""
        return "detect:merge"


# ---------------------------------------------------------------------------
# Candidate parsing (shared with detect.py)
# ---------------------------------------------------------------------------


def _to_candidates(
    resp: DetectorResponse,
    model: DetectorName,
    classes: dict[str, ClassConfig],
) -> list[Candidate]:
    """Map server-echoed labels back to canonical class names by prompt match.

    v4: classes is dict[str, ClassConfig] with .prompts (list of str).
    """
    prompt_to_class: dict[str, str] = {}
    for cls_name, cls_cfg in classes.items():
        for prompt in cls_cfg.prompts:
            prompt_to_class[prompt.lower().strip()] = cls_name

    out: list[Candidate] = []
    for i, (box, score, label) in enumerate(
        zip(resp.boxes, resp.scores, resp.labels)
    ):
        if len(box) != 4:
            continue
        cls_name = prompt_to_class.get(str(label).lower().strip())
        if cls_name is None:
            logger.debug(
                "%s: unrecognised label %r; dropped (known: %s)",
                model.value,
                label,
                sorted(prompt_to_class),
            )
            continue
        out.append(
            Candidate(
                candidate_id=f"{model.value}:{cls_name}:{i}",
                class_name=cls_name,
                label=str(label),
                source_model=model.value,
                expression=str(label),
                bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                score=float(score),
                status=CandidateStatus.PROPOSED,
            )
        )
    return out
