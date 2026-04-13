from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from ..config import ClassPackConfig
from ..contracts import Candidate, ReviewDecision
from ..prompts import build_verification_prompt
from ..registry import register_adapter
from ..utils import crop_candidate, draw_candidate, pil_to_data_url
from .base import AnnotationAdapter


def _extract_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"```json|```", "", clean).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(clean[start : end + 1])


@register_adapter("qwen")
class QwenAdapter(AnnotationAdapter):
    capabilities = {"verification"}

    def __init__(self, name, config):
        super().__init__(name, config)
        params = config.params
        self.base_url = params.get("base_url", "http://localhost:8005/v1").rstrip("/")
        self.api_key = params.get("api_key", "dummy")
        self.model_name = config.model_id or params.get("model", "Qwen/Qwen3.5-27B-FP8")

    def propose(self, image: Image.Image, class_pack: ClassPackConfig, expression: str, params: dict[str, Any]) -> list[Candidate]:
        raise NotImplementedError

    def refine(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any], request: ReviewDecision | None = None) -> Candidate | None:
        return None

    def verify(self, image: Image.Image, candidate: Candidate, class_pack: ClassPackConfig, params: dict[str, Any]) -> ReviewDecision:
        overlay = draw_candidate(image, candidate)
        crop = crop_candidate(image, candidate, padding=float(params.get("padding", 0.08)))
        prompt = build_verification_prompt(class_pack, candidate)
        payload = {
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": pil_to_data_url(overlay)}},
                        {"type": "image_url", "image_url": {"url": pil_to_data_url(crop)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        raw = self._post_with_retries(candidate, payload, params)
        if isinstance(raw, ReviewDecision):
            return raw
        content = raw["choices"][0]["message"]["content"]
        try:
            payload = _extract_json(content or "{}")
        except Exception as exc:
            return self._failure_decision(candidate, "invalid_json", f"Verifier returned invalid JSON: {exc}")
        try:
            return ReviewDecision.model_validate({"candidate_id": candidate.candidate_id, **payload})
        except Exception as exc:
            return self._failure_decision(candidate, "schema_error", f"Verifier output failed schema validation: {exc}")

    def _post_with_retries(self, candidate: Candidate, payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | ReviewDecision:
        max_retries = int(params.get("max_retries", 2))
        timeout = float(params.get("timeout", 120))
        base_backoff = float(params.get("retry_backoff_seconds", 1.0))

        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                url=f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
                retryable = exc.code >= 500 or exc.code in {408, 409, 429}
                if retryable and attempt < max_retries:
                    time.sleep(base_backoff * (2**attempt))
                    continue
                return self._failure_decision(candidate=candidate, failure_type="http_error", failure_reason=f"HTTP {exc.code}: {body or exc.reason}")
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                if attempt < max_retries:
                    time.sleep(base_backoff * (2**attempt))
                    continue
                failure_type = "timeout_error" if isinstance(exc, (socket.timeout, TimeoutError)) else "transport_error"
                return self._failure_decision(candidate=candidate, failure_type=failure_type, failure_reason=str(exc))

        return self._failure_decision(candidate=candidate, failure_type="transport_error", failure_reason="Verifier request exhausted retries.")

    def _failure_decision(self, candidate: Candidate | None, failure_type: str, failure_reason: str) -> ReviewDecision:
        candidate_id = candidate.candidate_id if candidate is not None else "unknown"
        return ReviewDecision(
            candidate_id=candidate_id,
            semantic_match="uncertain",
            bbox_tight="uncertain",
            recommended_action="escalate",
            confidence_band="low",
            rationale_short="Verification failed.",
            next_stage="escalation",
            failure_type=failure_type,
            failure_reason=failure_reason,
        )