"""Features the tier-2 rankers see for each candidate."""

from datetime import date
from typing import List, Optional

from hledger_ai.candidates.amounts import AmountCandidate
from hledger_ai.candidates.dates import DateCandidate, TimeCandidate
from hledger_ai.models.features import (
    amount_features,
    date_features,
    time_features,
    words,
)


def test_words_are_lowercase_plain_letters_without_numbers() -> None:
    assert words(text="TOTAAL € 11,54 Café-Nr 3") == ["totaal", "cafe", "nr"]


def _amount(
    value: float, row: int, row_text: str, previous: str = ""
) -> AmountCandidate:
    return AmountCandidate(
        value=value,
        row=row,
        row_count=10,
        row_text=row_text,
        previous_row_text=previous,
        relative_height=1.0,
        x_relative=0.8,
        is_percent=False,
        has_currency=False,
        is_last_in_row=True,
    )


CASH_RECEIPT: List[AmountCandidate] = [
    _amount(2.49, 2, "Brood 2,49"),
    _amount(9.05, 3, "Kaas 9,05"),
    _amount(11.54, 5, "TOTAAL 11,54"),
    _amount(20.00, 6, "CONTANT 20,00"),
    _amount(8.46, 7, "8,46", previous="RETOUR"),
    _amount(11.54, 9, "Totaal incl 11,54"),
]


def test_largest_amount_is_flagged() -> None:
    flags = [
        amount_features(candidate=c, candidates=CASH_RECEIPT)["is_max"]
        for c in CASH_RECEIPT
    ]

    assert flags == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_cash_given_is_flagged_as_sum_of_total_and_change() -> None:
    # 20,00 = 11,54 + 8,46: the pattern of money handed over.
    features = [
        amount_features(candidate=c, candidates=CASH_RECEIPT)
        for c in CASH_RECEIPT
    ]

    assert [f["is_sum_of_two"] for f in features] == [0, 0, 1, 1, 0, 1]
    # 2,49 + 9,05 = 11,54 too: an item total is also a sum, which is why
    # the difference flag below is needed to tell change from total.
    assert [f["is_difference"] for f in features] == [1, 1, 1, 0, 1, 1]


def test_amount_printed_twice_counts_its_repetitions() -> None:
    features = amount_features(
        candidate=CASH_RECEIPT[2], candidates=CASH_RECEIPT
    )

    assert features["repeats"] == 1.0
    assert (
        amount_features(candidate=CASH_RECEIPT[0], candidates=CASH_RECEIPT)[
            "repeats"
        ]
        == 0.0
    )


def test_words_on_the_row_become_features() -> None:
    features = amount_features(
        candidate=CASH_RECEIPT[3], candidates=CASH_RECEIPT
    )

    assert features["w:contant"] == 1.0
    assert "w:totaal" not in features


def test_amount_alone_on_its_row_borrows_the_words_above_it() -> None:
    features = amount_features(
        candidate=CASH_RECEIPT[4], candidates=CASH_RECEIPT
    )

    assert features["p:retour"] == 1.0


def test_round_amounts_are_flagged_as_typical_cash() -> None:
    assert (
        amount_features(candidate=CASH_RECEIPT[3], candidates=CASH_RECEIPT)[
            "is_round"
        ]
        == 1.0
    )
    assert (
        amount_features(candidate=CASH_RECEIPT[2], candidates=CASH_RECEIPT)[
            "is_round"
        ]
        == 0.0
    )


def _date(
    value: date, days_before_capture: Optional[int], source: str = "ocr"
) -> DateCandidate:
    return DateCandidate(
        value=value,
        style="dmy",
        source=source,
        row=1,
        row_count=10,
        row_text="Datum 17-03-2025 14:32",
        previous_row_text="",
        has_time_in_row=True,
        days_before_capture=days_before_capture,
    )


def test_distance_between_printed_date_and_photo_is_bucketed() -> None:
    cases = [
        (None, "cap:unknown"),
        (-288, "cap:future"),
        (0, "cap:same_day"),
        (2, "cap:1-3d"),
        (10, "cap:4-14d"),
        (40, "cap:15-60d"),
        (200, "cap:61-365d"),
        (800, "cap:over_1y"),
    ]
    for days, bucket in cases:
        candidate = _date(date(2025, 3, 17), days)
        features = date_features(candidate=candidate, candidates=[candidate])
        assert features.get(bucket) == 1.0, (days, bucket)


def test_date_features_tell_printed_dates_from_the_photo_day() -> None:
    printed = _date(date(2025, 3, 17), 2)
    photo = _date(date(2025, 3, 19), 0, source="capture")

    assert (
        date_features(candidate=printed, candidates=[printed, photo])[
            "source:ocr"
        ]
        == 1.0
    )
    assert (
        date_features(candidate=photo, candidates=[printed, photo])[
            "source:capture"
        ]
        == 1.0
    )


def test_time_next_to_a_date_is_flagged() -> None:
    beside = TimeCandidate(
        value="14:32",
        row=1,
        row_count=10,
        row_text="17-03-2025 14:32",
        has_date_in_row=True,
        has_seconds=False,
    )
    opening_hours = TimeCandidate(
        value="22:00",
        row=8,
        row_count=10,
        row_text="Open tot 22:00",
        has_date_in_row=False,
        has_seconds=False,
    )
    both = [beside, opening_hours]

    assert time_features(candidate=beside, candidates=both)["with_date"] == 1.0
    assert (
        time_features(candidate=opening_hours, candidates=both)["with_date"]
        == 0.0
    )
    assert (
        time_features(candidate=opening_hours, candidates=both)["w:open"] == 1.0
    )
