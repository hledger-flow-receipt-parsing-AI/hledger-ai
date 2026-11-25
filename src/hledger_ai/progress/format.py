"""Compact texts for progress, ETA and accuracy (UI-2)."""

from __future__ import annotations

from typing import Optional

from typeguard import typechecked


@typechecked
def format_eta(*, seconds: float) -> str:
    """``21m``, ``352h`` or ``34d``: minutes up to 99, hours up to 499."""
    minutes = seconds / 60.0
    if minutes < 100:
        return f"{max(1, round(minutes))}m"
    hours = minutes / 60.0
    if hours < 500:
        return f"{round(hours)}h"
    return f"{round(hours / 24.0)}d"


@typechecked
def format_percent(*, fraction: float) -> str:
    """Whole percent, rounded down so 100% only shows when really done."""
    return f"{int(fraction * 100 + 1e-9)}%"


@typechecked
def _stage(*, fraction: float, eta: Optional[float]) -> str:
    percent = format_percent(fraction=fraction)
    if fraction >= 1.0:
        return percent
    return f"{percent},{format_eta(seconds=eta) if eta is not None else '-'}"


@typechecked
def format_cell(
    *,
    training_fraction: float,
    training_eta: Optional[float],
    inference_fraction: float,
    inference_eta: Optional[float],
    accuracy: Optional[float],
) -> str:
    """The five values of one model/strategy cell:
    ``23%,352h 45%,21m 87%``."""
    return " ".join(
        [
            _stage(fraction=training_fraction, eta=training_eta),
            _stage(fraction=inference_fraction, eta=inference_eta),
            "-" if accuracy is None else format_percent(fraction=accuracy),
        ]
    )
