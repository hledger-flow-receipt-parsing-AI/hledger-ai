"""The training progress table (UI-1 to UI-6)."""

from typing import Dict, Optional

import pytest

from hledger_ai.dataset.configuration import DatasetConfiguration
from hledger_ai.progress.view import (
    accuracy_colour,
    build_table,
    decode_key,
    render_table,
    step_configuration,
)
from hledger_ai.training.state import ModelState, StrategyState

PRIVATE = DatasetConfiguration(
    dataset_hash="dd8c873a2f40aa",
    private=True,
    receipt_count=369,
    registered_at="2026-11-25T13:33:00",
)
PUBLIC = DatasetConfiguration(
    dataset_hash="0123456789abcd",
    private=False,
    receipt_count=10,
    registered_at="2026-11-24T09:00:00",
)


def _strategy(correct: int, total: int, done: int, of: int, seconds: float):
    return StrategyState(
        infer_done=done,
        infer_total=of,
        infer_seconds=seconds,
        folds={
            "0": {
                "shop": {
                    "correct": correct,
                    "total": total,
                    "baseline_correct": 0,
                }
            }
        },
    )


def _states() -> Dict[str, Optional[ModelState]]:
    running = ModelState(
        model="rapidocr-linear",
        dataset_hash=PRIVATE.dataset_hash,
        private=True,
        model_version="2",
        hyperparameters={},
        ocr_done=50,
        ocr_total=100,
        ocr_read=50,
        ocr_seconds=100.0,
        fits_done=1,
        fits_total=5,
        fit_seconds=10.0,
        strategies={"ST-12": _strategy(7, 10, 30, 120, 60.0)},
    )
    priors = ModelState(
        model="priors",
        dataset_hash=PRIVATE.dataset_hash,
        private=True,
        model_version="2",
        hyperparameters={},
        fits_done=5,
        fits_total=5,
        fit_seconds=4.0,
        finished=True,
        strategies={"ST-1": _strategy(6, 10, 120, 120, 1.0)},
    )
    return {"priors": priors, "rapidocr-linear": running, "qwen-lora": None}


def _table():
    return build_table(
        configuration=PRIVATE,
        index=1,
        count=2,
        states=_states(),
        strategy_ids=["ST-1", "ST-12"],
    )


def test_cells_hold_the_five_values_per_model_and_strategy() -> None:
    table = _table()
    cells = {name: [c.text for c in row] for name, row in table.rows}

    # Training 110 s done, 140 s left; inference 30 of 120 at 2 s each.
    assert cells["rapidocr-linear"] == ["-", "44%,2m 25%,3m 70%"]
    # Tier-1 priors: rebuilt in seconds, so simply done (UI-6).
    assert cells["priors"] == ["100% 100% 60%", "-"]
    assert cells["qwen-lora"] == ["-", "-"]


def test_rows_are_models_and_columns_are_strategies() -> None:
    text = render_table(table=_table(), colour=False, width=200)
    lines = text.splitlines()
    header = next(line for line in lines if "ST-1" in line)

    assert header.index("ST-1") < header.index("ST-12")
    assert [
        line.split()[0]
        for line in lines
        if line.startswith(("priors", "rapidocr-linear", "qwen-lora"))
    ] == ["priors", "rapidocr-linear", "qwen-lora"]
    assert "\x1b[" not in text


def test_title_names_the_dataset_configuration_and_its_place_in_time() -> None:
    text = render_table(table=_table(), colour=False, width=200)

    assert "dataset 2/2" in text
    assert "dd8c873a2f40" in text
    assert "369 receipts" in text
    assert "private" in text
    assert "2026-11-25T13:33:00" in text


def test_private_configuration_is_highlighted_orange() -> None:
    orange = "\x1b[38;5;208m"
    private = render_table(table=_table(), colour=True, width=200)
    public = render_table(
        table=build_table(
            configuration=PUBLIC,
            index=0,
            count=2,
            states={},
            strategy_ids=["ST-1"],
        ),
        colour=True,
        width=200,
    )

    assert orange in private
    assert orange not in public
    assert "public" in public


def test_cell_background_goes_from_red_to_green_with_accuracy() -> None:
    low, middle, high = (accuracy_colour(accuracy=a) for a in (0.0, 0.5, 1.0))

    assert low[0] > low[1]  # red dominates
    assert high[1] > high[0]  # green dominates
    assert low[1] < middle[1] <= high[1]  # green rises with accuracy
    assert low[0] >= middle[0] > high[0]  # red falls with accuracy


def test_scored_cells_carry_a_background_and_unscored_ones_do_not() -> None:
    text = render_table(table=_table(), colour=True, width=200)
    linear = next(l for l in text.splitlines() if "rapidocr-linear" in l)
    idle = next(l for l in text.splitlines() if "qwen-lora" in l)

    assert "\x1b[48;2;" in linear
    assert "\x1b[48;2;" not in idle


@pytest.mark.parametrize(
    "index, key, want",
    [
        (0, "ctrl right", 1),
        (2, "ctrl right", 0),
        (0, "ctrl left", 2),
        (1, "ctrl left", 0),
        (1, "x", 1),
    ],
)
def test_ctrl_arrows_step_through_configurations_and_wrap(
    index: int, key: str, want: int
) -> None:
    assert step_configuration(index=index, count=3, key=key) == want


def test_stepping_without_configurations_stays_put() -> None:
    assert step_configuration(index=0, count=0, key="ctrl right") == 0


@pytest.mark.parametrize(
    "raw, want",
    [
        (b"\x1b[1;5D", "ctrl left"),
        (b"\x1b[1;5C", "ctrl right"),
        (b"\x1b[5D", "ctrl left"),  # rxvt
        (b"\x1b[5C", "ctrl right"),
        (b"\x1b[D", "left"),
        (b"\x1b[C", "right"),
        (b"q", "q"),
        (b"\x03", "q"),  # ctrl+c
        (b"", None),
    ],
)
def test_terminal_key_sequences_are_recognised(raw: bytes, want) -> None:
    assert decode_key(raw=raw) == want


def test_narrow_terminal_shows_the_columns_that_fit_and_says_so() -> None:
    table = build_table(
        configuration=PRIVATE,
        index=0,
        count=1,
        states=_states(),
        strategy_ids=["ST-1", "ST-12", "ST-13", "ST-14"],
    )

    narrow = render_table(table=table, colour=False, width=70, first_column=1)
    header = next(line for line in narrow.splitlines() if "ST-12" in line)

    assert "ST-1 " not in header
    assert "ST-14" not in header
    assert "columns 2-" in narrow and "of 4" in narrow
