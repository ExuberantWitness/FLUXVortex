"""Corrected scientific observers (refactor plan §14 U0-F).

These fix the reporting-truth defects catalogued in ANALYSIS_BRIEF_v4 §1.4.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch


def mean_map_then_max(z_history: torch.Tensor, window: slice) -> float:
    """Paper displacement metric: max of the time-MEAN camber map.

    z_history: (T, nc, ns+1) membrane height field.
    window: slice selecting the statistics window rows.
    Returns max(mean_map) — the metric the paper actually reports
    (time-averaged DIC, then maximum of the mean shape).
    """
    mean_map = z_history[window].mean(dim=0)
    return float(mean_map.max().item())


def mean_cn_over_window(cn_series: torch.Tensor, window: slice) -> float:
    """Same-window time-average of Cn (not an end-point)."""
    return float(cn_series[window].mean().item())


@dataclass(frozen=True)
class SignCrossingResult:
    station: tuple[int, int]         # (chord_index, span_index)
    crossings: int                    # count of sign changes in the window
    min_value: float
    max_value: float

def sign_crossing_at_station(
    z_history: torch.Tensor, window: slice,
    chord_index: int, span_index: int,
) -> SignCrossingResult:
    """Detect sign crossings at a SPECIFIED station — NOT via max(z) which
    can hide negative excursions when another region is positive."""
    series = z_history[window, chord_index, span_index]
    signs = torch.sign(series)
    changes = int((signs[1:] * signs[:-1] < 0).sum().item())
    return SignCrossingResult(
        station=(chord_index, span_index),
        crossings=changes,
        min_value=float(series.min().item()),
        max_value=float(series.max().item()),
    )


@dataclass(frozen=True)
class StatisticsWindowRecord:
    start_time_star: float
    end_time_star: float
    sample_count: int
    full_slow_mode_periods_covered: float  # window_length / slow_mode_period if known

def serialize_statistics_window(
    time_star: torch.Tensor, window: slice,
    slow_mode_period: float | None = None,
) -> StatisticsWindowRecord:
    """Serialize the ACTUAL window used — not the intended one."""
    ts = time_star[window]
    length = float(ts[-1] - ts[0])
    return StatisticsWindowRecord(
        start_time_star=float(ts[0].item()),
        end_time_star=float(ts[-1].item()),
        sample_count=int(ts.numel()),
        full_slow_mode_periods_covered=(
            length / slow_mode_period if slow_mode_period else -1.0
        ),
    )
