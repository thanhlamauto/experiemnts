"""Kaggle-first SiT-XL/2 protocol for miniImageNet analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ProtocolConfig

if TYPE_CHECKING:
    from .stages import AnalysisRuntime


def run_bootstrap_stage(*args: Any, **kwargs: Any):
    from .stages import run_bootstrap_stage as _run_bootstrap_stage

    return _run_bootstrap_stage(*args, **kwargs)


def run_task0_stage(*args: Any, **kwargs: Any):
    from .stages import run_task0_stage as _run_task0_stage

    return _run_task0_stage(*args, **kwargs)


def run_analysis_stage(*args: Any, **kwargs: Any):
    from .stages import run_analysis_stage as _run_analysis_stage

    return _run_analysis_stage(*args, **kwargs)


def create_analysis_runtime(*args: Any, **kwargs: Any):
    from .stages import create_analysis_runtime as _create_analysis_runtime

    return _create_analysis_runtime(*args, **kwargs)


__all__ = [
    "ProtocolConfig",
    "create_analysis_runtime",
    "run_analysis_stage",
    "run_bootstrap_stage",
    "run_task0_stage",
]
