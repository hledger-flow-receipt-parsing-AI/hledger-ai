"""Suggestions for the labeller's manual address questions: read from the
receipt, and the usual city and country. Invented shops only."""

from typing import Dict, List, Optional

import pytest
from helpers import make_evidence, make_label

from hledger_ai.address_parts import (
    QUESTION_OF_PART,
    AddressKnowledge,
    receipt_values,
    suggest_address_parts,
)
from hledger_ai.models.experts import TrainingExample
from hledger_ai.receipt_fields import facts_from_label
from hledger_ai.suggest.engine import Suggestion

HOME = "Voorbeeldstad"


def _example(
    *,
    shop: str,
    street: str,
    nr: str,
    zipcode: Optional[str],
    city: str = HOME,
    country: str = "Netherlands",
    rows: Optional[List[tuple]] = None,
) -> TrainingExample:
    label = make_label(
        shop_name=shop,
        street=street,
        house_nr=nr,
        zipcode=zipcode,
        city=city,
        country=country,
    )
    return TrainingExample(
        facts=facts_from_label(label=label),
        evidence=make_evidence(rows=rows or []),
    )


def _printed(shop: str, street: str, nr: str, zipcode: str, city: str) -> list:
    return [(shop.upper(),), (f"{street} {nr}",), (f"{zipcode} {city}",)]


def _examples() -> List[TrainingExample]:
    bakery = dict(shop="Bakkerij Voorbeeld", street="Dorpsstraat", nr="4")
    found = []
    for _ in range(3):
        found.append(
            _example(
                **bakery,
                zipcode="9999ZZ",
                rows=_printed(
                    "Bakkerij Voorbeeld", "Dorpsstraat", "4", "9999 ZZ", HOME
                ),
            )
        )
    found.append(
        _example(
            shop="Fietsenmaker Proef",
            street="Kerkplein",
            nr="7",
            zipcode="9997XX",
            rows=_printed(
                "Fietsenmaker Proef", "Kerkplein", "7", "9997XX", HOME
            ),
        )
    )
    # A receipt whose printed street is not the labelled one: a miss.
    found.append(
        _example(
            shop="Markt De Kraam",
            street="Plein",
            nr="1",
            zipcode="9996WW",
            city="Ergensveen",
            rows=_printed(
                "Markt De Kraam", "Achterom", "9", "9996 WW", "Ergensveen"
            ),
        )
    )
    found.append(
        _example(
            shop="Beispiel Laden",
            street="Hauptstrasse",
            nr="5",
            zipcode="12345",
            city="Beispielstadt",
            country="Germany",
        )
    )
    return found


def _values(found: Dict[str, List[Suggestion]], question: str) -> List[str]:
    return [s.value for s in found.get(question, [])]


def test_questions_are_those_of_the_labeller() -> None:
    assert QUESTION_OF_PART == {
        "shop_name": "shop_name",
        "street": "shop_street",
        "house_nr": "shop_house_nr",
        "zipcode": "shop_zipcode",
        "city": "shop_city",
        "country": "shop_country",
    }


def test_read_precision_is_measured_per_shop() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    # Three shops have a photo; the bakery's three receipts weigh one shop.
    # The street is right for two of them (Achterom is not Plein).
    assert knowledge.read_counts(part="street", rank=0) == (2.0, 3.0)
    assert knowledge.read_counts(part="zipcode", rank=0) == (3.0, 3.0)
    assert knowledge.read_precision(part="street", rank=0) == (2 + 1) / (3 + 2)
    assert knowledge.read_precision(part="shop_name", rank=2) is None


def test_a_receipt_never_borrows_its_own_shops_spelling() -> None:
    """Measured as for a shop seen for the first time: the OCR slip
    "PR0EFJE" may not be corrected with the shop's own label, also when
    that shop was labelled once with "9995VV" and once with "9995 vv"."""
    found = [
        _example(
            shop="Proefje",
            street="Laan",
            nr="1",
            zipcode=zipcode,
            rows=[("PR0EFJE",), ("Laan 1",), (f"{zipcode} {HOME}",)],
        )
        for zipcode in ("9995VV", "9995 vv")
    ]
    knowledge = AddressKnowledge.fit(examples=found)
    assert knowledge.read_counts(part="shop_name", rank=0) == (0.0, 1.0)


def test_a_new_shop_gets_what_is_printed_on_its_receipt() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    receipt = make_evidence(
        rows=_printed(
            "Nieuwe Winkel", "Molenweg", "21", "9995 VV", "Ergensveen"
        )
    )
    found = suggest_address_parts(
        knowledge=knowledge, evidence=receipt, answers={}
    )
    assert _values(found, "shop_zipcode")[0] == "9995VV"
    assert _values(found, "shop_street")[0] == "Molenweg"
    assert _values(found, "shop_house_nr")[0] == "21"
    assert _values(found, "shop_name")[0] == "Nieuwe Winkel"
    best = found["shop_street"][0]
    assert best.confidence == knowledge.read_precision(part="street", rank=0)
    assert best.sources == ("ST-12",)


def test_a_known_city_keeps_its_labelled_spelling() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    receipt = make_evidence(rows=[("Winkel",), ("9995 VV ERGENSVEEN",)])
    found = suggest_address_parts(
        knowledge=knowledge, evidence=receipt, answers={}
    )
    assert _values(found, "shop_city")[0] == "Ergensveen"


def test_the_usual_city_and_country_are_suggested_without_a_photo() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    found = suggest_address_parts(
        knowledge=knowledge, evidence=None, answers={}
    )
    # Every shop counts once: two of the four shops are in Voorbeeldstad.
    assert _values(found, "shop_city")[0] == HOME
    assert found["shop_city"][0].confidence == 2 / 4
    assert _values(found, "shop_country") == ["Netherlands", "Germany"]
    assert found["shop_country"][0].sources == ("ST-19",)
    # The known shops' streets are never offered: "manual address" says
    # the shop is none of them.
    assert "shop_street" not in found and "shop_name" not in found


def test_a_dutch_zipcode_on_the_receipt_points_at_the_country() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    receipt = make_evidence(rows=[("Winkel",), ("9995 VV Ergensveen",)])
    found = suggest_address_parts(
        knowledge=knowledge, evidence=receipt, answers={}
    )
    best = found["shop_country"][0]
    assert best.value == "Netherlands"
    # Every Dutch-looking labelled zipcode was in the Netherlands, times
    # how often the zipcode read is right.
    assert best.confidence == pytest.approx(
        knowledge.read_precision(part="zipcode", rank=0)
    )


def test_a_typed_part_is_never_suggested_again() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    receipt = make_evidence(
        rows=_printed(
            "Nieuwe Winkel", "Molenweg", "21", "9995 VV", "Ergensveen"
        )
    )
    found = suggest_address_parts(
        knowledge=knowledge,
        evidence=receipt,
        answers={"shop_name": "Nieuwe Winkel", "shop_street": "Molenweg"},
    )
    assert "shop_name" not in found and "shop_street" not in found
    assert _values(found, "shop_zipcode")[0] == "9995VV"


def test_a_near_known_spelling_never_replaces_the_read() -> None:
    """Hoofdstraat is no misread Hoogstraat; a one-letter slip of a known
    street is offered in its known spelling too, after what was read."""
    knowledge = AddressKnowledge.fit(
        examples=[
            _example(
                shop="Ander", street="Hoogstraat", nr="2", zipcode="9990AA"
            ),
            _example(
                shop="Derde", street="Dorpsstraat", nr="3", zipcode="9991AA"
            ),
        ]
    )
    known = list(knowledge.spellings["street"].values())
    values = receipt_values(
        evidence=make_evidence(rows=[("Winkel",), ("Hoofdstraat 5",)]),
        part="street",
        known=known,
    )
    assert values == ["Hoofdstraat"]
    slip = receipt_values(
        evidence=make_evidence(rows=[("Winkel",), ("Dorpsstraal 5",)]),
        part="street",
        known=known,
    )
    assert slip == ["Dorpsstraal", "Dorpsstraat"]


def test_old_engines_without_address_knowledge_suggest_nothing() -> None:
    assert (
        suggest_address_parts(knowledge=None, evidence=None, answers={}) == {}
    )


def test_knowledge_survives_saving() -> None:
    knowledge = AddressKnowledge.fit(examples=_examples())
    again = AddressKnowledge.from_json(data=knowledge.to_json())
    assert again == knowledge
