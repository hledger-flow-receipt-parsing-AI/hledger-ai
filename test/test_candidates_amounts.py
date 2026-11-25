"""Money amounts found in OCR text."""

from typing import List

import pytest

from hledger_ai.candidates.amounts import amount_candidates, find_amounts
from hledger_ai.ocr.types import OcrLine, OcrResult


@pytest.mark.parametrize(
    "text, want",
    [
        ("TOTAAL 11,54", [11.54]),
        ("TOTAL 11.54", [11.54]),
        ("€ 1.234,56", [1234.56]),
        ("USD 1,234.56", [1234.56]),
        ("12 ,34", [12.34]),  # OCR often splits before the comma
        ("12, 34", [12.34]),
        ("Brood 2,49 Kaas 5,10", [2.49, 5.10]),
        ("KORTING -1,50", [-1.50]),
        ("0,395 kg", []),  # three decimals: a weight, not money
        ("17-03-2025 14:32", []),
        ("17.03.2025", []),  # a date written with dots
        ("tel 0612345678", []),
        ("1234 AB Voorbeeldstad", []),
    ],
)
def test_amounts_in_text(text: str, want: List[float]) -> None:
    assert [a.value for a in find_amounts(text=text)] == want


def test_percentages_are_marked_so_a_tax_rate_is_not_read_as_money() -> None:
    found = find_amounts(text="BTW 9,00% 0,95")

    assert [(a.value, a.is_percent) for a in found] == [
        (9.0, True),
        (0.95, False),
    ]


def test_currency_sign_next_to_the_amount_is_noticed() -> None:
    found = find_amounts(text="€11,54 en 3,00")

    assert [(a.value, a.has_currency) for a in found] == [
        (11.54, True),
        (3.0, False),
    ]


def _box(text: str, x: float, y: float, h: float = 30) -> OcrLine:
    return OcrLine(
        text=text,
        score=0.99,
        box=((x, y), (x + 100, y), (x + 100, y + h), (x, y + h)),
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


def test_candidates_know_their_row_and_the_words_beside_them() -> None:
    result = _result(
        [
            _box("Brood", 40, 100),
            _box("2,49", 400, 100),
            _box("TOTAAL", 40, 200, h=45),
            _box("2,49", 400, 200, h=45),
        ]
    )

    found = amount_candidates(results=[result])

    assert [(c.value, c.row, c.row_text) for c in found] == [
        (2.49, 0, "Brood 2,49"),
        (2.49, 1, "TOTAAL 2,49"),
    ]
    assert found[1].previous_row_text == "Brood 2,49"
    assert found[0].row_count == 2
    # The total is printed larger than the item line.
    assert found[1].relative_height > found[0].relative_height


def test_rows_of_a_second_photo_continue_after_the_first() -> None:
    first = _result([_box("Brood", 40, 100), _box("2,49", 400, 100)])
    second = _result([_box("TOTAAL", 40, 50), _box("2,49", 400, 50)])

    found = amount_candidates(results=[first, second])

    assert [(c.row, c.row_text) for c in found] == [
        (0, "Brood 2,49"),
        (1, "TOTAAL 2,49"),
    ]
