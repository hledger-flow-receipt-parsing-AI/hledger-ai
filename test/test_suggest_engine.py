"""The suggestion engine: first guesses plus cross-inference (CI-1..CI-4)."""

from pathlib import Path
from typing import List

import pytest
from helpers import card_receipt, cash_receipt, make_evidence, make_label

from hledger_ai.models.experts import TrainingExample
from hledger_ai.receipt_fields import facts_from_label
from hledger_ai.suggest.engine import SuggestionEngine

BAKERY = "Winkel: Bakkerstraat, 1, 1111 AA, Voorbeeldstad, Netherlands"
HARDWARE = "Winkel: IJzerweg, 9, 2222 BB, Voorbeeldstad, Netherlands"


def _example(
    *, category: str, street: str, house_nr: str, zipcode: str, bank: str
) -> TrainingExample:
    """Both shops print the very same receipt text ("Winkel")."""
    return TrainingExample(
        facts=facts_from_label(
            label=make_label(
                category=category,
                shop_name="Winkel",
                street=street,
                house_nr=house_nr,
                zipcode=zipcode,
                bank=bank,
                account_type=(
                    "checking" if bank == "examplebank" else "physical"
                ),
                amount=5.0,
                change=0.0 if bank == "examplebank" else 15.0,
            )
        ),
        evidence=make_evidence(
            rows=[("Winkel",), ("Datum 17-03-2025 14:32",), ("TOTAAL", "5,00")]
        ),
    )


def _examples() -> List[TrainingExample]:
    bakery = [
        _example(
            category="food:bakery",
            street="Bakkerstraat",
            house_nr="1",
            zipcode="1111 AA",
            bank="examplebank",
        )
        for _ in range(6)
    ]
    hardware = [
        _example(
            category="tools",
            street="IJzerweg",
            house_nr="9",
            zipcode="2222 BB",
            bank="wallet",
        )
        for _ in range(4)
    ]
    return bakery + hardware


RECEIPT = make_evidence(
    rows=[("Winkel",), ("Datum 20-03-2025 10:00",), ("TOTAAL", "7,50")]
)


def test_first_guess_is_a_ranked_list_with_confidences_per_field() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    suggestions = engine.suggest(evidence=RECEIPT, known={})

    assert suggestions["amount"][0].value == 7.5
    for field in ("date", "time", "amount", "category", "shop", "bank_account"):
        confidences = [s.confidence for s in suggestions[field]]
        assert confidences == sorted(confidences, reverse=True), field
        assert all(0.0 <= c <= 1.0 for c in confidences), field


def test_confirmed_fields_are_never_suggested_again() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    suggestions = engine.suggest(
        evidence=RECEIPT, known={"category": "tools", "amount": 7.5}
    )

    assert "category" not in suggestions
    assert "amount" not in suggestions


def test_confirming_the_category_reranks_the_shops() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    blind = engine.suggest(evidence=RECEIPT, known={})["shop"]
    informed = engine.suggest(evidence=RECEIPT, known={"category": "tools"})[
        "shop"
    ]

    # The text cannot tell the two shops apart; the bakery is more frequent.
    assert blind[0].value == BAKERY
    assert informed[0].value == HARDWARE
    assert informed[0].sources == ("ST-12", "ST-1")
    assert blind[0].sources == ("ST-12",)


def test_unseen_subcategory_still_helps_through_its_parent() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    informed = engine.suggest(
        evidence=RECEIPT, known={"category": "tools:garden"}
    )["shop"]

    assert informed[0].value == HARDWARE


def test_known_value_that_was_never_seen_leaves_the_image_ranking() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    informed = engine.suggest(evidence=RECEIPT, known={"category": "travel"})

    assert informed["shop"][0].value == BAKERY


def test_priors_alone_work_without_any_image() -> None:
    engine = SuggestionEngine.train(examples=_examples(), use_image=False)

    suggestions = engine.suggest(evidence=None, known={"category": "tools"})

    assert suggestions["shop"][0].value == HARDWARE
    assert suggestions["shop"][0].sources == ("ST-1",)
    assert suggestions["bank_account"][0].value == "alex:wallet:physical"
    # Nothing known that says anything about the amount.
    assert "amount" not in suggestions


def test_strategy_filter_isolates_one_strategy() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    only_image = engine.suggest(
        evidence=RECEIPT, known={"category": "tools"}, strategies={"ST-12"}
    )

    assert only_image["shop"][0].value == BAKERY
    assert list(only_image) == ["shop"]


def _card_and_cash_examples() -> List[TrainingExample]:
    examples = []
    for total in ("11,54", "7,20", "23,05", "4,99", "15,75", "31,10"):
        paid = float(total.replace(",", "."))
        examples.append(
            TrainingExample(
                facts=facts_from_label(
                    label=make_label(amount=paid, change=0.0)
                ),
                evidence=card_receipt(total=total),
            )
        )
        examples.append(
            TrainingExample(
                facts=facts_from_label(
                    label=make_label(
                        amount=50.0,
                        change=round(50.0 - paid, 2),
                        bank="wallet",
                        account_type="physical",
                        category="food:market",
                    )
                ),
                evidence=cash_receipt(
                    total=total,
                    given="50,00",
                    back=f"{50.0 - paid:.2f}".replace(".", ","),
                ),
            )
        )
    return examples


def test_known_payment_account_decides_whether_change_is_expected() -> None:
    engine = SuggestionEngine.train(examples=_card_and_cash_examples() * 3)
    # Nothing on this receipt says how it was paid.
    unclear = make_evidence(rows=[("TOTAAL", "6,35"), ("13,65",)])

    by_card = engine.suggest(
        evidence=unclear, known={"bank_account": "alex:examplebank:checking"}
    )
    in_cash = engine.suggest(
        evidence=unclear, known={"bank_account": "alex:wallet:physical"}
    )

    # In the labels card payments never returned change, cash always did.
    assert by_card["change"][0].value == 0.0
    assert in_cash["change"][0].value > 0.0
    assert by_card["change"][0].sources == ("ST-11", "ST-17")


def test_engine_gives_the_same_answers_after_saving_and_loading(
    tmp_path: Path,
) -> None:
    engine = SuggestionEngine.train(examples=_examples())
    path = str(tmp_path / "engine.joblib")

    engine.save(path=path)
    restored = SuggestionEngine.load(path=path)

    before = engine.suggest(evidence=RECEIPT, known={"category": "tools"})
    after = restored.suggest(evidence=RECEIPT, known={"category": "tools"})
    assert [(s.value, s.sources) for s in after["shop"]] == [
        (s.value, s.sources) for s in before["shop"]
    ]
    assert [s.confidence for s in after["shop"]] == pytest.approx(
        [s.confidence for s in before["shop"]]
    )


def test_engine_trained_on_nothing_suggests_nothing() -> None:
    engine = SuggestionEngine.train(examples=[])

    assert engine.suggest(evidence=RECEIPT, known={}) == {}


def test_real_looking_receipts_get_amount_and_change_right() -> None:
    engine = SuggestionEngine.train(examples=_card_and_cash_examples())

    cash = engine.suggest(
        evidence=cash_receipt(total="6,35", given="20,00", back="13,65"),
        known={},
    )

    assert cash["amount"][0].value == 20.0
    assert cash["change"][0].value == 13.65
    assert cash["bank_account"][0].value == "alex:wallet:physical"


def test_known_amount_handed_over_points_at_the_matching_change() -> None:
    # Two amounts could be the change; only one fits 20,00 - 6,35.
    engine = SuggestionEngine.train(examples=_card_and_cash_examples())
    receipt = make_evidence(
        rows=[
            ("Markt De Kraam",),
            ("Kaas", "6,35"),
            ("TOTAAL", "6,35"),
            ("CONTANT", "20,00"),
            ("RETOUR", "13,65"),
            ("RETOUR EMBALLAGE", "0,25"),
        ]
    )

    informed = engine.suggest(evidence=receipt, known={"amount": 20.0})

    assert informed["change"][0].value == 13.65
    assert "ST-18" in informed["change"][0].sources


def test_only_the_requested_fields_are_worked_out() -> None:
    engine = SuggestionEngine.train(examples=_examples())

    suggestions = engine.suggest(
        evidence=RECEIPT, known={}, fields={"amount", "shop"}
    )

    assert sorted(suggestions) == ["amount", "shop"]


def test_with_nothing_known_the_priors_fall_back_to_the_usual_values() -> None:
    engine = SuggestionEngine.train(examples=_examples(), use_image=False)

    suggestions = engine.suggest(evidence=None, known={})

    # Six of the ten receipts were paid from the bank account.
    assert suggestions["bank_account"][0].value == "alex:examplebank:checking"
    assert suggestions["bank_account"][0].confidence == pytest.approx(0.6)
    assert suggestions["bank_account"][0].sources == ("ST-19",)
    assert suggestions["currency"][0].value == "EUR"
    assert suggestions["change"][0].value == 0.0
    assert "amount" not in suggestions
