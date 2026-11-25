"""Where on the photo the evidence for a suggestion is (UI-7).

The labelling TUI draws thin boxes around it, so the user checks a
suggestion at a glance instead of searching the receipt. Amounts, dates
and times point at the OCR boxes that hold the suggested value; class
fields (payment account, category, shop, ...) point at the boxes whose
text pushed the model towards its answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Sequence, Tuple

from typeguard import typechecked

from hledger_ai.candidates.amounts import find_amounts
from hledger_ai.candidates.dates import find_dates, find_times
from hledger_ai.models.experts import ChangeExpert, ClassExpert
from hledger_ai.ocr.types import OcrLine, Point
from hledger_ai.suggest.engine import Suggestion, SuggestionEngine
from hledger_ai.suggest.evidence import ReceiptEvidence

Box = Tuple[Point, Point, Point, Point]
# A class is explained by at most this many boxes, each at least this share
# of the strongest one.
MAX_EVIDENCE_BOXES = 3
MIN_EVIDENCE_SHARE = 0.3
_AMOUNT_FIELDS = ("amount", "subtotal", "total_tax")
_CLASS_FIELDS = (
    "category",
    "bank_account",
    "currency",
    "is_withdrawal",
    "shop",
)


@typechecked
@dataclass(frozen=True)
class Highlight:
    image_path: str  # the photo the box is on
    box: Box  # its four corners, in that photo's pixels
    text: str  # what was read there


@typechecked
@dataclass(frozen=True)
class LocatedRow:
    image_path: str
    lines: Tuple[OcrLine, ...]

    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


@typechecked
def located_rows(
    *, evidence: ReceiptEvidence, image_paths: Sequence[str]
) -> List[LocatedRow]:
    """The printed rows of every photo, in the order candidates count them."""
    rows: List[LocatedRow] = []
    for result, path in zip(evidence.ocr, image_paths):
        rows += [
            LocatedRow(image_path=path, lines=tuple(row))
            for row in result.rows()
        ]
    return rows


@typechecked
def _cents(*, value: float) -> int:
    return int(round(value * 100))


@typechecked
def _boxes_holding(
    *, rows: Sequence[LocatedRow], holds: Callable[[str], bool]
) -> List[Highlight]:
    """Boxes whose text *holds* the value; when only a whole row does (the
    value is split over boxes), every box of that row."""
    found: List[Highlight] = []
    for row in rows:
        hits = [line for line in row.lines if holds(line.text)]
        if not hits and len(row.lines) > 1 and holds(row.text()):
            hits = list(row.lines)
        found += [
            Highlight(image_path=row.image_path, box=line.box, text=line.text)
            for line in hits
        ]
    return found


@typechecked
def amount_highlights(
    *, rows: Sequence[LocatedRow], value: float
) -> List[Highlight]:
    cents = _cents(value=value)
    return _boxes_holding(
        rows=rows,
        holds=lambda text: any(
            _cents(value=found.value) == cents
            for found in find_amounts(text=text)
        ),
    )


@typechecked
def date_highlights(
    *, rows: Sequence[LocatedRow], value: date
) -> List[Highlight]:
    return _boxes_holding(
        rows=rows,
        holds=lambda text: any(
            found.value == value for found in find_dates(text=text)
        ),
    )


@typechecked
def time_highlights(
    *, rows: Sequence[LocatedRow], value: str
) -> List[Highlight]:
    return _boxes_holding(
        rows=rows,
        holds=lambda text: any(
            found.value == value for found in find_times(text=text)
        ),
    )


@typechecked
def evidence_highlights(
    *,
    rows: Sequence[LocatedRow],
    scores: Callable[[List[str]], List[float]],
    limit: int = MAX_EVIDENCE_BOXES,
    share: float = MIN_EVIDENCE_SHARE,
) -> List[Highlight]:
    """The boxes whose text speaks most for a class (*scores* rates texts):
    the strongest few, in reading order."""
    boxes = [(row.image_path, line) for row in rows for line in row.lines]
    if not boxes:
        return []
    rated = sorted(
        zip(scores([line.text for _, line in boxes]), range(len(boxes))),
        key=lambda pair: (-pair[0], pair[1]),
    )
    best = rated[0][0]
    if best <= 0.0:
        return []
    chosen = sorted(
        index for score, index in rated[:limit] if score >= share * best
    )
    return [
        Highlight(
            image_path=boxes[i][0], box=boxes[i][1].box, text=boxes[i][1].text
        )
        for i in chosen
    ]


@typechecked
def _highlights_for(
    *,
    engine: SuggestionEngine,
    rows: Sequence[LocatedRow],
    field: str,
    value: object,
) -> List[Highlight]:
    if field in _AMOUNT_FIELDS and isinstance(value, (int, float)):
        return amount_highlights(rows=rows, value=float(value))
    if field == "change" and isinstance(value, (int, float)):
        if value > 0:
            return amount_highlights(rows=rows, value=float(value))
        expert = engine.expert(field="change")
        if not isinstance(expert, ChangeExpert):
            return []
        return evidence_highlights(
            rows=rows,
            scores=lambda texts: expert.text_scores(
                texts=texts, has_change=False
            ),
        )
    if field == "date" and isinstance(value, date):
        return date_highlights(rows=rows, value=value)
    if field == "time" and isinstance(value, str):
        return time_highlights(rows=rows, value=value)
    if field in _CLASS_FIELDS:
        expert = engine.expert(field=field)
        if not isinstance(expert, ClassExpert):
            return []
        return evidence_highlights(
            rows=rows,
            scores=lambda texts: expert.text_scores(
                texts=texts, value=str(value)
            ),
        )
    return []


@typechecked
def field_highlights(
    *,
    engine: SuggestionEngine,
    evidence: ReceiptEvidence,
    image_paths: Sequence[str],
    suggestions: Dict[str, List[Suggestion]],
) -> Dict[str, List[Highlight]]:
    """Per field, the boxes behind its first suggestion; fields without
    any are left out."""
    rows = located_rows(evidence=evidence, image_paths=image_paths)
    found: Dict[str, List[Highlight]] = {}
    for field, ranked in suggestions.items():
        if not ranked:
            continue
        boxes = _highlights_for(
            engine=engine, rows=rows, field=field, value=ranked[0].value
        )
        if boxes:
            found[field] = boxes
    return found
