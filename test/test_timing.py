"""Field timestamps while labelling (SP-4) and strategy time budgets (SP-5)."""

import json
from pathlib import Path
from typing import List

import pytest

from hledger_ai.timing import FieldTimer, strategy_budgets


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _events(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_first_answer_is_entered_and_a_changed_answer_is_re_edited(
    tmp_path: Path,
) -> None:
    clock = Clock()
    timer = FieldTimer(
        path=str(tmp_path / "t.jsonl"), receipt_id="r1", clock=clock
    )

    timer.answered(question_id="category", answer="food")
    clock.now += 5
    timer.answered(question_id="category", answer="food")  # unchanged
    clock.now += 5
    timer.answered(question_id="category", answer="tools")

    events = _events(tmp_path / "t.jsonl")
    assert [(e["question_id"], e["event"], e["time"]) for e in events] == [
        ("category", "entered", 1000.0),
        ("category", "re-edited", 1010.0),
    ]
    assert all(e["receipt_id"] == "r1" for e in events)
    # The answers themselves are private and not needed for timing.
    assert all("answer" not in e for e in events)


def test_reaching_a_question_is_recorded_once_per_visit(tmp_path: Path) -> None:
    clock = Clock()
    timer = FieldTimer(
        path=str(tmp_path / "t.jsonl"), receipt_id="r1", clock=clock
    )

    timer.reached(question_id="address_selector")
    timer.reached(question_id="address_selector")  # still there
    timer.reached(question_id="subtotal")
    timer.reached(question_id="address_selector")  # came back

    assert [e["question_id"] for e in _events(tmp_path / "t.jsonl")] == [
        "address_selector",
        "subtotal",
        "address_selector",
    ]


def _log(path: Path, receipt: str, rows: List[tuple]) -> None:
    with open(path, "a") as handle:
        for question_id, event, moment in rows:
            handle.write(
                json.dumps(
                    {
                        "receipt_id": receipt,
                        "question_id": question_id,
                        "event": event,
                        "time": moment,
                    }
                )
                + "\n"
            )


def test_budget_is_the_typical_time_from_known_field_to_improved_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.jsonl"
    for receipt, gap in (("r1", 20.0), ("r2", 23.0), ("r3", 40.0)):
        _log(
            path,
            receipt,
            [
                ("category", "entered", 100.0),
                ("address_selector", "reached", 100.0 + gap),
            ],
        )

    budget = strategy_budgets(timings_path=str(path), default=5.0)

    # ST-1: category -> shop (the address selector). Median of 20, 23, 40.
    assert budget("ST-1") == pytest.approx(23.0)


def test_re_edit_restarts_the_clock_for_that_receipt(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _log(
        path,
        "r1",
        [
            ("category", "entered", 100.0),
            ("category", "re-edited", 115.0),
            ("address_selector", "reached", 125.0),
        ],
    )

    budget = strategy_budgets(timings_path=str(path), default=5.0)

    assert budget("ST-1") == pytest.approx(10.0)


def test_strategy_with_two_known_fields_counts_from_the_last_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.jsonl"
    _log(
        path,
        "r1",
        [
            ("amount", "entered", 100.0),
            ("change", "entered", 104.0),
            ("subtotal", "reached", 130.0),
            ("total_tax", "reached", 140.0),
        ],
    )

    budget = strategy_budgets(timings_path=str(path), default=5.0)

    # ST-20 improves subtotal and tax; the first one reached sets the limit.
    assert budget("ST-20") == pytest.approx(26.0)


def test_without_timings_the_default_budget_applies(tmp_path: Path) -> None:
    budget = strategy_budgets(
        timings_path=str(tmp_path / "missing.jsonl"), default=23.0
    )

    assert budget("ST-1") == 23.0
    assert budget("ST-12") == 23.0  # image only: nothing to wait for
