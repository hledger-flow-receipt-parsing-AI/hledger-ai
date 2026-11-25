"""Money amounts found in the OCR text of a receipt.

Every amount on the receipt is a candidate for the paid amount, the change,
the subtotal or the tax; a trained ranker picks among them.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Sequence

from typeguard import typechecked

from hledger_ai.ocr.types import OcrResult

_AMOUNT = re.compile(
    r"""
    (?<![\d.,])                              # not inside a longer number
    (?P<sign>-\s?)?
    (?P<whole>\d{1,3}(?:[.,]\d{3})+|\d+)     # 1.234 or 1,234 or 1234
    \s?[.,]\s?                               # OCR may split at the separator
    (?P<cents>\d{2})
    (?![.,]?\d)                              # dates, weights (0,395 kg)
    """,
    re.VERBOSE,
)
_CURRENCY = re.compile(r"[€$£]|\b(?:EUR|USD|GBP|CHF|PLN|SEK|NOK|DKK|CZK)\b")


@typechecked
@dataclass(frozen=True)
class FoundAmount:
    value: float
    is_percent: bool
    has_currency: bool
    start: int
    end: int


@typechecked
def find_amounts(*, text: str) -> List[FoundAmount]:
    """Amounts with two decimals in *text*, in reading order."""
    found: List[FoundAmount] = []
    for match in _AMOUNT.finditer(text):
        whole = re.sub(r"[.,]", "", match.group("whole"))
        value = float(f"{whole}.{match.group('cents')}")
        if match.group("sign"):
            value = -value
        before = text[max(0, match.start() - 5) : match.start()]
        after = text[match.end() : match.end() + 5]
        found.append(
            FoundAmount(
                value=value,
                is_percent=after.lstrip().startswith("%"),
                has_currency=bool(
                    _CURRENCY.search(before) or _CURRENCY.search(after)
                ),
                start=match.start(),
                end=match.end(),
            )
        )
    return found


@typechecked
@dataclass(frozen=True)
class AmountCandidate:
    """One amount occurrence with what a ranker needs to judge it."""

    value: float
    row: int  # row number over all photos of the receipt
    row_count: int
    row_text: str
    previous_row_text: str
    relative_height: float  # box height against the photo's typical box
    x_relative: float  # horizontal centre of the box, 0 (left) to 1
    is_percent: bool
    has_currency: bool
    is_last_in_row: bool


@typechecked
def amount_candidates(*, results: Sequence[OcrResult]) -> List[AmountCandidate]:
    """All amount occurrences of a receipt (one OCR result per photo)."""
    rows = []  # (row boxes, typical height, image width)
    for result in results:
        heights = [line.height() for line in result.lines]
        typical = statistics.median(heights) if heights else 1.0
        for row in result.rows():
            rows.append((row, typical or 1.0, max(result.width, 1)))

    candidates: List[AmountCandidate] = []
    for index, (row, typical, width) in enumerate(rows):
        row_text = " ".join(line.text for line in row)
        previous = (
            " ".join(line.text for line in rows[index - 1][0])
            if index > 0
            else ""
        )
        in_row: List[AmountCandidate] = []
        for line in row:
            for amount in find_amounts(text=line.text):
                in_row.append(
                    AmountCandidate(
                        value=amount.value,
                        row=index,
                        row_count=len(rows),
                        row_text=row_text,
                        previous_row_text=previous,
                        relative_height=line.height() / typical,
                        x_relative=line.centre()[0] / width,
                        is_percent=amount.is_percent,
                        has_currency=amount.has_currency,
                        is_last_in_row=False,
                    )
                )
        if in_row:
            last = in_row[-1]
            in_row[-1] = AmountCandidate(
                **{**last.__dict__, "is_last_in_row": True}
            )
        candidates.extend(in_row)
    return candidates
