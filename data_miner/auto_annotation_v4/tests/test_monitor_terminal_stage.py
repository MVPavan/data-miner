"""Unit tests for PipelineMonitor terminal-stage computation."""

from __future__ import annotations

from data_miner.auto_annotation_v4.configs.enums import Stage
from data_miner.auto_annotation_v4.workers.monitor import (
    PipelineMonitor,
    _compute_terminal_stage,
)


class _StubDB:
    """Stand-in for CheckpointDB — PipelineMonitor only stores the reference."""


def test_terminal_stage_detect_filter():
    assert _compute_terminal_stage([Stage.DETECT, Stage.FILTER]) is Stage.FILTER


def test_terminal_stage_full_pipeline():
    stages = [
        Stage.DETECT,
        Stage.FILTER,
        Stage.EVALUATE,
        Stage.REFINE,
        Stage.FINALIZE,
    ]
    assert _compute_terminal_stage(stages) is Stage.FINALIZE


def test_terminal_stage_evaluate_refine_finalize():
    assert (
        _compute_terminal_stage([Stage.EVALUATE, Stage.REFINE, Stage.FINALIZE])
        is Stage.FINALIZE
    )


def test_terminal_stage_out_of_order_uses_stage_order():
    # REFINE should win over EVALUATE regardless of list order.
    assert (
        _compute_terminal_stage([Stage.REFINE, Stage.DETECT, Stage.EVALUATE])
        is Stage.REFINE
    )


def test_terminal_stage_empty_defaults_to_finalize():
    assert _compute_terminal_stage([]) is Stage.FINALIZE
    assert _compute_terminal_stage(None) is Stage.FINALIZE


def test_monitor_sets_terminal_stage_from_ctor_arg():
    db = _StubDB()
    mon = PipelineMonitor(db, stages=[Stage.DETECT, Stage.FILTER])  # type: ignore[arg-type]
    assert mon.terminal_stage is Stage.FILTER

    mon2 = PipelineMonitor(db)  # type: ignore[arg-type]
    assert mon2.terminal_stage is Stage.FINALIZE
