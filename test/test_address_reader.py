"""Reading the shop's name and address from the receipt text, for the
labeller's manual address questions. Invented receipts only."""

from typing import Dict, List

import pytest
from helpers import make_evidence

from hledger_ai.address_reader import comparable_part, read_address


def _values(rows: List[tuple]) -> Dict[str, List[str]]:
    found = read_address(evidence=make_evidence(rows=rows))
    return {part: [r.value for r in reads] for part, reads in found.items()}


HEADER = [
    ("VOORBEELD MARKT",),
    ("Dorpsstraat 12",),
    ("1234 AB VOORBEELDSTAD",),
    ("Tel 012-3456789",),
    ("Brood", "2,49"),
    ("TOTAAL", "2,49"),
]


def test_a_dutch_header_gives_every_part() -> None:
    found = _values(HEADER)
    assert found["zipcode"][0] == "1234AB"  # as the labels write it
    assert found["city"][0] == "Voorbeeldstad"
    assert found["street"][0] == "Dorpsstraat"
    assert found["house_nr"][0] == "12"
    assert found["shop_name"][0] == "Voorbeeld Markt"


def test_street_and_zipcode_on_one_row() -> None:
    found = _values([("Bakkerij Proef",), ("Kerkplein 7, 9997XX Testdorp",)])
    assert found["street"][0] == "Kerkplein"
    assert found["house_nr"][0] == "7"
    assert found["zipcode"][0] == "9997XX"
    assert found["city"][0] == "Testdorp"


def test_a_foreign_zipcode_and_city() -> None:
    found = _values(
        [("Beispiel Laden",), ("Hauptstrasse 5",), ("12345 Beispielstadt",)]
    )
    assert found["zipcode"][0] == "12345"
    assert found["city"][0] == "Beispielstadt"
    assert found["street"][0] == "Hauptstrasse"


def test_the_city_on_the_row_under_the_zipcode() -> None:
    found = _values([("Winkel",), ("Laan 3a",), ("9996 WW",), ("ERGENSVEEN",)])
    assert found["zipcode"][0] == "9996WW"
    assert found["city"][0] == "Ergensveen"
    assert found["house_nr"][0] == "3a"


def test_units_and_receipt_words_are_not_read_as_an_address() -> None:
    found = _values(
        [
            ("Kassa 3",),
            ("Bon 1234",),
            ("Melk 1500 ML", "1,19"),
            ("Datum 17-03-2025 14:32",),
            ("PIN", "1,19"),
        ]
    )
    assert found.get("zipcode", []) == []
    assert found.get("street", []) == []


def test_welcome_words_are_not_part_of_the_name() -> None:
    found = _values([("Welkom bij",), ("Groente De Hoek",), ("Markt 1",)])
    assert found["shop_name"][0] == "Groente De Hoek"


def test_a_receipt_without_text_reads_nothing() -> None:
    assert read_address(evidence=make_evidence(rows=[])) == {}


@pytest.mark.parametrize(
    "part, left, right",
    [
        ("zipcode", "1234 ab", "1234AB"),
        ("street", "Sint-Jansstraat", "Sint Jansstraat"),
        ("city", "Café", "CAFE"),
    ],
)
def test_parts_compare_without_case_spaces_or_accents(
    part: str, left: str, right: str
) -> None:
    assert comparable_part(part=part, value=left) == comparable_part(
        part=part, value=right
    )


def test_a_shop_name_keeps_to_the_name_rule() -> None:
    """Letters, digits and " &'-./" only: a ":" would also break the
    address list, which writes "name: address"."""
    found = _values([("DE BAKKER: VERS (CENTRUM)",), ("Markt 1",)])
    name = found["shop_name"][0]
    assert name == "De Bakker Vers Centrum"
    assert all(c.isalnum() or c in " &'-./" for c in name)
    assert _values([("MCDONALD'S",)])["shop_name"][0] == "Mcdonald's"


@pytest.mark.parametrize(
    "row",
    ["Bonnummer 1234", "Kassanr 3", "Medewerker 12", "Postbus 1234"],
)
def test_numbered_receipt_words_are_no_street(row: str) -> None:
    assert _values([("Winkel",), (row,)]).get("street", []) == []


def test_a_phone_number_after_the_city_is_left_out() -> None:
    found = _values([("Winkel",), ("1234 AB Voorbeeldstad Tel 012-3456789",)])
    assert found["zipcode"][0] == "1234AB"
    assert found["city"][0] == "Voorbeeldstad"


def test_initials_with_a_dot_are_no_web_address() -> None:
    found = _values([("Slagerij J.de Vries",), ("Markt 1",)])
    assert found["shop_name"][0] == "Slagerij J.de Vries"


def test_a_year_is_no_zipcode() -> None:
    found = _values([("Winkel",), ("SINDS 1923 AMBACHTELIJK",)])
    assert found.get("zipcode", []) == []
