"""Compact progress texts (UI-2)."""

from typing import Optional

import pytest

from hledger_ai.progress.format import format_cell, format_eta, format_percent


@pytest.mark.parametrize(
    "seconds, want",
    [
        (30, "1m"),  # never "0m" while something is left
        (140, "2m"),
        (21 * 60, "21m"),
        (99 * 60, "99m"),
        (100 * 60, "2h"),
        (352 * 3600, "352h"),
        (499 * 3600, "499h"),
        (500 * 3600, "21d"),
        (34 * 86400, "34d"),
    ],
)
def test_eta_is_shown_in_minutes_hours_or_days(
    seconds: float, want: str
) -> None:
    assert format_eta(seconds=seconds) == want


@pytest.mark.parametrize(
    "fraction, want",
    [(0.0, "0%"), (0.234, "23%"), (0.999, "99%"), (1.0, "100%")],
)
def test_percent_never_claims_done_before_it_is(
    fraction: float, want: str
) -> None:
    assert format_percent(fraction=fraction) == want


def test_cell_shows_training_inference_and_accuracy() -> None:
    cell = format_cell(
        training_fraction=0.23,
        training_eta=352 * 3600.0,
        inference_fraction=0.45,
        inference_eta=21 * 60.0,
        accuracy=0.87,
    )

    assert cell == "23%,352h 45%,21m 87%"


def test_finished_stages_show_no_eta() -> None:
    cell = format_cell(
        training_fraction=1.0,
        training_eta=0.0,
        inference_fraction=1.0,
        inference_eta=0.0,
        accuracy=0.874,
    )

    assert cell == "100% 100% 87%"


def test_values_not_available_yet_show_a_dash() -> None:
    cell = format_cell(
        training_fraction=0.0,
        training_eta=None,
        inference_fraction=0.0,
        inference_eta=None,
        accuracy=None,
    )

    assert cell == "0%,- 0%,- -"
