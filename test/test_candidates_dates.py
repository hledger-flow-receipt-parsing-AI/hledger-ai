"""Dates and times found in OCR text."""

from datetime import date, datetime
from typing import List, Optional, Tuple

import pytest

from hledger_ai.candidates.dates import (
    capture_time_from_filename,
    date_candidates,
    find_dates,
    find_times,
    time_candidates,
)
from hledger_ai.ocr.types import OcrLine, OcrResult

D = date(2025, 3, 17)


@pytest.mark.parametrize(
    "text, want",
    [
        ("17-03-2025", [(D, "dmy")]),
        ("17/03/25", [(D, "dmy")]),
        ("17.03.2025", [(D, "dmy")]),
        ("17 - 03 - 2025", [(D, "dmy")]),  # OCR spaces
        ("2025-03-17", [(D, "ymd")]),
        ("Datum: 17-03-202514:32", [(D, "dmy")]),  # time glued to the year
        ("Datum:17-03-2025", [(D, "dmy")]),
        ("03/17/2025", [(D, "mdy")]),  # 17 cannot be a month
        ("17 mrt 2025", [(D, "text")]),
        ("17 maart 2025", [(D, "text")]),
        ("17-MAR-25", [(D, "text")]),
        ("Mar 17, 2025", [(D, "text")]),
        ("March 17 2025", [(D, "text")]),
        ("32-13-2025", []),
        ("17-03-1925", []),  # not a plausible receipt year
        ("1234 AB Voorbeeldstad", []),
        ("TOTAAL 11,54", []),
        ("BTW 21% 1.234,56", []),
    ],
)
def test_dates_in_text(text: str, want: List[Tuple[date, str]]) -> None:
    assert [(d.value, d.style) for d in find_dates(text=text)] == want


def test_ambiguous_day_month_order_yields_both_readings() -> None:
    found = find_dates(text="05/03/2025")

    assert [(d.value, d.style) for d in found] == [
        (date(2025, 3, 5), "dmy"),
        (date(2025, 5, 3), "mdy"),
    ]


@pytest.mark.parametrize(
    "text, want",
    [
        ("14:32", ["14:32"]),
        ("14:32:07", ["14:32"]),
        ("9:05", ["09:05"]),
        ("17-03-2025 14:32", ["14:32"]),
        ("17-03-202514:32", ["14:32"]),
        ("2:32 PM", ["14:32"]),
        ("12:05 AM", ["00:05"]),
        ("Time:10:30", ["10:30"]),  # no space after the label's colon
        ("Tijd:14:32:07", ["14:32"]),
        ("14.32 uur", ["14:32"]),
        ("14.32.07", ["14:32"]),
        ("25:99", []),
        ("TOTAAL 11,54", []),
        ("14.32", []),  # without a cue this is more likely an amount
        ("1234 AB", []),
    ],
)
def test_times_in_text(text: str, want: List[str]) -> None:
    assert [t.value for t in find_times(text=text)] == want


def test_dotted_time_is_accepted_next_to_a_date() -> None:
    assert [t.value for t in find_times(text="17.03.2025 14.32")] == ["14:32"]


@pytest.mark.parametrize(
    "path, want",
    [
        ("/in/20250317_143207.jpg", datetime(2025, 3, 17, 14, 32, 7)),
        ("/in/IMG_20250317_143207.jpg", datetime(2025, 3, 17, 14, 32, 7)),
        ("/p/20250317_143207_cropped.jpg", datetime(2025, 3, 17, 14, 32, 7)),
        ("/in/scan0001.jpg", None),
        ("/in/20259999_143207.jpg", None),
    ],
)
def test_capture_time_from_photo_filename(
    path: str, want: Optional[datetime]
) -> None:
    assert capture_time_from_filename(path=path) == want


def _box(text: str, x: float, y: float) -> OcrLine:
    return OcrLine(
        text=text,
        score=0.99,
        box=((x, y), (x + 100, y), (x + 100, y + 30), (x, y + 30)),
    )


def _result(lines: List[OcrLine]) -> OcrResult:
    return OcrResult(
        engine="fake",
        image_sha256="0" * 64,
        width=600,
        height=900,
        elapsed_s=0.0,
        lines=tuple(lines),
    )


def test_date_candidates_carry_row_context_and_distance_to_the_photo() -> None:
    result = _result(
        [
            _box("Datum", 40, 100),
            _box("17-03-2025", 200, 100),
            _box("14:32", 400, 100),
            _box("Geldig tot 01-01-2026", 40, 300),
        ]
    )

    found = date_candidates(
        results=[result], captured_at=datetime(2025, 3, 19, 8, 0)
    )
    printed = [c for c in found if c.source == "ocr"]

    assert [(c.value, c.row_text) for c in printed] == [
        (D, "Datum 17-03-2025 14:32"),
        (date(2026, 1, 1), "Geldig tot 01-01-2026"),
    ]
    assert [c.days_before_capture for c in printed] == [2, -288]
    assert [c.has_time_in_row for c in printed] == [True, False]


def test_photo_date_is_offered_as_a_fallback_candidate() -> None:
    found = date_candidates(
        results=[_result([_box("TOTAAL 11,54", 40, 100)])],
        captured_at=datetime(2025, 3, 19, 8, 0),
    )

    assert [(c.value, c.source) for c in found] == [
        (date(2025, 3, 19), "capture")
    ]


def test_without_photo_time_and_without_printed_date_there_is_nothing() -> None:
    found = date_candidates(
        results=[_result([_box("TOTAAL 11,54", 40, 100)])], captured_at=None
    )

    assert found == []


def test_time_candidates_know_whether_a_date_shares_their_row() -> None:
    result = _result(
        [
            _box("17-03-2025", 40, 100),
            _box("14:32", 400, 100),
            _box("Open tot 22:00", 40, 300),
        ]
    )

    found = time_candidates(results=[result])

    assert [(c.value, c.has_date_in_row) for c in found] == [
        ("14:32", True),
        ("22:00", False),
    ]
