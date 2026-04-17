"""Tier 1 unit tests for auto_annotation_v4 enums."""

from __future__ import annotations

from data_miner.auto_annotation_v4.configs.enums import STAGE_ORDER, Stage


def test_stage_order_has_filter():
    assert STAGE_ORDER == [
        Stage.DETECT,
        Stage.FILTER,
        Stage.EVALUATE,
        Stage.REFINE,
        Stage.FINALIZE,
    ]
