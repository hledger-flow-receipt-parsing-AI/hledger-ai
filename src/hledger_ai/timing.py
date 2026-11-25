"""When each field is answered while labelling (SP-4), and from that the
time a model has to improve a later field (SP-5).

Only question ids and timestamps are stored, never the answers.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from typeguard import typechecked

from hledger_ai.strategies import get_strategy

# Receipt field -> id of the labeller question that asks for it.
FIELD_QUESTION: Dict[str, str] = {
    "date": "date",
    "time": "date",
    "is_withdrawal": "is_withdrawal",
    "category": "category",
    "bank_account": "bank_account",
    "currency": "currency",
    "amount": "amount",
    "change": "change",
    "shop": "address_selector",
    "subtotal": "subtotal",
    "total_tax": "total_tax",
    "shop_name": "shop_name",
    "address": "shop_zipcode",
}


@typechecked
class FieldTimer:
    """Appends one line per event of one receipt to the timings file."""

    def __init__(
        self,
        *,
        path: str,
        receipt_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._receipt_id = receipt_id
        self._clock = clock
        self._answers: Dict[str, Any] = {}
        self._at: Optional[str] = None

    def _write(self, *, question_id: str, event: str) -> None:
        os.makedirs(
            os.path.dirname(self._path) or ".", mode=0o700, exist_ok=True
        )
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "receipt_id": self._receipt_id,
                        "question_id": question_id,
                        "event": event,
                        "time": self._clock(),
                    }
                )
                + "\n"
            )

    def answered(self, *, question_id: str, answer: Any) -> None:
        """Record a first answer ("entered") or a changed one
        ("re-edited"); an unchanged answer is not an event."""
        if question_id not in self._answers:
            self._write(question_id=question_id, event="entered")
        elif self._answers[question_id] != answer:
            self._write(question_id=question_id, event="re-edited")
        self._answers[question_id] = answer

    def reached(self, *, question_id: str) -> None:
        """Record that the cursor arrived at a question."""
        if question_id != self._at:
            self._write(question_id=question_id, event="reached")
        self._at = question_id


@typechecked
def _read(*, path: str) -> Dict[str, List[Tuple[str, str, float]]]:
    """receipt id -> [(question id, event, time)] in file order."""
    by_receipt: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    if not os.path.isfile(path):
        return by_receipt
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                by_receipt[row["receipt_id"]].append(
                    (row["question_id"], row["event"], float(row["time"]))
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return by_receipt


@typechecked
def _gap(
    *,
    events: List[Tuple[str, str, float]],
    known: List[str],
    improved: List[str],
) -> Optional[float]:
    """Seconds from the last answer to a known question until the first
    improved question was reached after it."""
    answered_at: Dict[str, float] = {}
    for question_id, event, moment in events:
        if event in ("entered", "re-edited") and question_id in known:
            answered_at[question_id] = moment
    if set(answered_at) != set(known):
        return None
    start = max(answered_at.values())
    reached = [
        moment
        for question_id, event, moment in events
        if event == "reached" and question_id in improved and moment > start
    ]
    return min(reached) - start if reached else None


@typechecked
def strategy_budgets(
    *, timings_path: str, default: float
) -> Callable[[str], float]:
    """SP-5: seconds a strategy may take, per strategy id: the median time
    between confirming its known fields and reaching a field it improves.
    *default* applies while there are no timings for it."""
    by_receipt = _read(path=timings_path)

    def budget(strategy_id: str) -> float:
        strategy = get_strategy(strategy_id=strategy_id)
        known = sorted(
            {FIELD_QUESTION[f] for f in strategy.known if f in FIELD_QUESTION}
        )
        improved = sorted(
            {
                FIELD_QUESTION[f]
                for f in strategy.improves
                if f in FIELD_QUESTION
            }
        )
        if not known or not improved:
            return default
        gaps = [
            gap
            for events in by_receipt.values()
            if (gap := _gap(events=events, known=known, improved=improved))
            is not None
        ]
        return statistics.median(gaps) if gaps else default

    return budget
