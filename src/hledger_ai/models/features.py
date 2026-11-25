"""Features of one candidate, as seen by a tier-2 ranker.

The words printed beside a candidate become features, so the rankers learn
the user's own receipts ("totaal", "contant", "retour", ...) from the labels
instead of from a fixed keyword list.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence

from typeguard import typechecked

from hledger_ai.candidates.amounts import AmountCandidate
from hledger_ai.candidates.dates import DateCandidate, TimeCandidate

_WORD = re.compile(r"[a-z]{2,}")
_MAX_WORDS = 12


@typechecked
def words(*, text: str) -> List[str]:
    """Lowercase letter runs without accents, digits or punctuation."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return _WORD.findall(plain)[:_MAX_WORDS]


@typechecked
def _context(*, row_text: str, previous_row_text: str) -> Dict[str, float]:
    """Words on the row; a row without words borrows the row above."""
    features: Dict[str, float] = {}
    row_words = words(text=row_text)
    for word in row_words:
        features[f"w:{word}"] = 1.0
    if not row_words:
        for word in words(text=previous_row_text):
            features[f"p:{word}"] = 1.0
    return features


@typechecked
def _cents(*, value: float) -> int:
    return int(round(value * 100))


@typechecked
def amount_features(
    *, candidate: AmountCandidate, candidates: Sequence[AmountCandidate]
) -> Dict[str, float]:
    money = [c for c in candidates if not c.is_percent and c.value > 0]
    values = sorted({_cents(value=c.value) for c in money})
    cents = _cents(value=candidate.value)
    occurrences = sum(1 for c in money if _cents(value=c.value) == cents)
    others = [v for v in values if v != cents]
    is_sum = any(
        (cents - v) in others and (cents - v) != v for v in others if v < cents
    )
    is_difference = any((cents + v) in values for v in others)
    rank = values[::-1].index(cents) if cents in values else len(values)

    features: Dict[str, float] = {
        "position": candidate.row / max(candidate.row_count - 1, 1),
        "x": candidate.x_relative,
        "height": min(max(candidate.relative_height, 0.3), 3.0),
        "is_percent": float(candidate.is_percent),
        "has_currency": float(candidate.has_currency),
        "is_last_in_row": float(candidate.is_last_in_row),
        "is_negative": float(candidate.value < 0),
        "is_max": float(bool(values) and cents == values[-1]),
        "is_second": float(rank == 1),
        "rank": rank / max(len(values), 1),
        "repeats": float(occurrences >= 2),
        "repeats_many": float(occurrences >= 3),
        "is_sum_of_two": float(is_sum),
        "is_difference": float(is_difference),
        "is_round": float(cents % 100 == 0),
        "is_round_five": float(cents % 500 == 0),
    }
    features.update(
        _context(
            row_text=candidate.row_text,
            previous_row_text=candidate.previous_row_text,
        )
    )
    return features


@typechecked
def _capture_bucket(*, days: Optional[int]) -> str:
    if days is None:
        return "cap:unknown"
    if days < 0:
        return "cap:future"
    if days == 0:
        return "cap:same_day"
    for limit, name in ((3, "1-3d"), (14, "4-14d"), (60, "15-60d")):
        if days <= limit:
            return f"cap:{name}"
    return "cap:61-365d" if days <= 365 else "cap:over_1y"


@typechecked
def date_features(
    *, candidate: DateCandidate, candidates: Sequence[DateCandidate]
) -> Dict[str, float]:
    printed = [c.value for c in candidates if c.source == "ocr"]
    occurrences = sum(1 for value in printed if value == candidate.value)
    features: Dict[str, float] = {
        f"style:{candidate.style}": 1.0,
        f"source:{candidate.source}": 1.0,
        _capture_bucket(days=candidate.days_before_capture): 1.0,
        "position": candidate.row / max(candidate.row_count - 1, 1),
        "with_time": float(candidate.has_time_in_row),
        "repeats": float(occurrences >= 2),
        "is_latest": float(bool(printed) and candidate.value == max(printed)),
        "is_earliest": float(bool(printed) and candidate.value == min(printed)),
        "is_only_printed": float(len(set(printed)) == 1),
        "no_printed_date": float(not printed),
    }
    features.update(
        _context(
            row_text=candidate.row_text,
            previous_row_text=candidate.previous_row_text,
        )
    )
    return features


@typechecked
def time_features(
    *, candidate: TimeCandidate, candidates: Sequence[TimeCandidate]
) -> Dict[str, float]:
    occurrences = sum(1 for c in candidates if c.value == candidate.value)
    features: Dict[str, float] = {
        "position": candidate.row / max(candidate.row_count - 1, 1),
        "with_date": float(candidate.has_date_in_row),
        "has_seconds": float(candidate.has_seconds),
        "repeats": float(occurrences >= 2),
        "is_only": float(len({c.value for c in candidates}) == 1),
        "is_first": float(candidate.row == min(c.row for c in candidates)),
        "is_last": float(candidate.row == max(c.row for c in candidates)),
    }
    features.update(_context(row_text=candidate.row_text, previous_row_text=""))
    return features
