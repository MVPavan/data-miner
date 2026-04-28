"""YAML loader + typed models for the label-validation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Thresholds:
    positive_thr: float = 0.25
    negative_thr: float = 0.25
    pos_neg_margin_thr: float = 0.05
    junk_thr: float = 0.30
    pos_junk_margin_thr: float = 0.05


@dataclass
class ClassPrompts:
    """All prompts + thresholds for a single class, with common junk already merged."""
    name: str
    positive: list[str]
    negative: list[str]
    junk: list[str]
    thr: Thresholds


@dataclass
class ValidationConfig:
    model: str
    classes_to_check: list[str]
    prompts: dict[str, ClassPrompts] = field(default_factory=dict)

    def prompt_order(self, class_name: str) -> tuple[list[str], int, int, int]:
        """Flat prompt order used when scoring: positives, then negatives, then junk.

        Returns (all_prompts, n_pos, n_neg, n_junk).
        """
        cp = self.prompts[class_name]
        all_prompts = list(cp.positive) + list(cp.negative) + list(cp.junk)
        return all_prompts, len(cp.positive), len(cp.negative), len(cp.junk)


def _resolve_thresholds(defaults: dict, override: Optional[dict]) -> Thresholds:
    merged = dict(defaults)
    if override:
        merged.update({k: v for k, v in override.items() if k in
                       {"positive_thr", "negative_thr", "pos_neg_margin_thr",
                        "junk_thr", "pos_junk_margin_thr"}})
    return Thresholds(**merged)


def load_config(path: str | Path) -> ValidationConfig:
    """Load and validate a label-validation YAML.

    Raises ``ValueError`` on any schema problem (unknown class, empty
    positive list, etc.) so the runner fails fast before loading a model.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    model = data.get("model", "siglip2-giant")
    classes_to_check: list[str] = list(data.get("classes_to_check") or [])
    if not classes_to_check:
        raise ValueError(f"{path}: classes_to_check is empty")

    defaults = {
        "positive_thr": 0.25,
        "negative_thr": 0.25,
        "pos_neg_margin_thr": 0.05,
        "junk_thr": 0.30,
        "pos_junk_margin_thr": 0.05,
    }
    defaults.update(data.get("defaults") or {})

    common_junk: list[str] = list(data.get("common_junk") or [])
    classes_block: dict = data.get("classes") or {}

    prompts: dict[str, ClassPrompts] = {}
    for name in classes_to_check:
        if name not in classes_block:
            raise ValueError(
                f"{path}: class {name!r} is in classes_to_check but has no "
                f"entry under classes:"
            )
        cb = classes_block[name] or {}
        positive = list(cb.get("positive") or [])
        negative = list(cb.get("negative") or [])
        # Per-class junk plus common_junk, de-duped, order-preserving.
        junk_raw = list(cb.get("junk") or []) + common_junk
        seen: set[str] = set()
        junk: list[str] = []
        for p in junk_raw:
            if p not in seen:
                seen.add(p)
                junk.append(p)
        if not positive:
            raise ValueError(
                f"{path}: class {name!r} has no positive prompts"
            )
        prompts[name] = ClassPrompts(
            name=name,
            positive=positive,
            negative=negative,
            junk=junk,
            thr=_resolve_thresholds(defaults, cb),
        )

    return ValidationConfig(
        model=model,
        classes_to_check=classes_to_check,
        prompts=prompts,
    )
