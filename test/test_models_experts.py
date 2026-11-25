"""Tier-2 experts: a first guess per field from the receipt image (CI-2)."""

from datetime import date
from typing import List

from helpers import card_receipt, cash_receipt, make_evidence, make_label

from hledger_ai.models.experts import (
    AmountExpert,
    ChangeExpert,
    ClassExpert,
    DateExpert,
    TimeExpert,
    TrainingExample,
)
from hledger_ai.receipt_fields import facts_from_label

TOTALS = ["11,54", "7,20", "23,05", "4,99", "15,75", "31,10", "8,40", "12,00"]


def _card_examples() -> List[TrainingExample]:
    return [
        TrainingExample(
            facts=facts_from_label(
                label=make_label(
                    amount=float(total.replace(",", ".")),
                    change=0.0,
                    total_tax=0.95,
                    category="food:bakery",
                    bank="examplebank",
                )
            ),
            evidence=card_receipt(total=total),
        )
        for total in TOTALS
    ]


def _cash_examples() -> List[TrainingExample]:
    examples = []
    for total in TOTALS:
        paid = float(total.replace(",", "."))
        given = 50.0
        back = round(given - paid, 2)
        examples.append(
            TrainingExample(
                facts=facts_from_label(
                    label=make_label(
                        date="2025-03-18T09:05:00",
                        amount=given,
                        change=back,
                        category="food:market",
                        bank="wallet",
                        account_type="physical",
                        shop_name="Markt De Kraam",
                    )
                ),
                evidence=cash_receipt(
                    total=total,
                    given="50,00",
                    back=f"{back:.2f}".replace(".", ","),
                ),
            )
        )
    return examples


def test_amount_expert_learns_that_the_total_row_holds_the_amount() -> None:
    expert = AmountExpert(field="amount").fit(examples=_card_examples())

    ranked = expert.suggest(evidence=card_receipt(total="6,35"), known={})

    # 99,99 (the loyalty balance) is the largest number on the receipt.
    assert ranked[0][0] == 6.35
    assert ranked[0][1] > 0.5


def test_tax_expert_picks_the_tax_row_not_the_total() -> None:
    expert = AmountExpert(field="total_tax").fit(examples=_card_examples())

    ranked = expert.suggest(evidence=card_receipt(total="6,35"), known={})

    assert ranked[0][0] == 0.95


def test_amount_expert_without_readable_amounts_suggests_nothing() -> None:
    expert = AmountExpert(field="amount").fit(examples=_card_examples())

    assert (
        expert.suggest(evidence=make_evidence(rows=[("Bedankt",)]), known={})
        == []
    )


def test_change_expert_says_zero_for_card_and_the_returned_cash_for_cash() -> (
    None
):
    expert = ChangeExpert().fit(examples=_card_examples() + _cash_examples())

    card = expert.suggest(evidence=card_receipt(total="6,35"), known={})
    cash = expert.suggest(
        evidence=cash_receipt(total="6,35", given="20,00", back="13,65"),
        known={},
    )

    assert card[0][0] == 0.0
    assert cash[0][0] == 13.65


def test_cash_receipts_teach_the_amount_expert_to_take_what_was_handed_over() -> (
    None
):
    expert = AmountExpert(field="amount").fit(examples=_cash_examples())

    ranked = expert.suggest(
        evidence=cash_receipt(total="6,35", given="20,00", back="13,65"),
        known={},
    )

    assert ranked[0][0] == 20.0


def test_date_expert_prefers_the_purchase_date_over_other_printed_dates() -> (
    None
):
    expert = DateExpert().fit(examples=_card_examples())
    receipt = make_evidence(
        rows=[
            ("Voorbeeld Markt",),
            ("Datum 25-06-2025 10:11",),
            ("TOTAAL", "3,00"),
            ("Ruilen tot 30-06-2025",),
        ]
    )

    ranked = expert.suggest(evidence=receipt, known={})

    assert ranked[0][0] == date(2025, 6, 25)
    assert date(2025, 6, 30) in [value for value, _ in ranked]


def test_date_expert_learns_the_day_month_order_from_the_labels() -> None:
    # "05-03-2025" reads as 5 March or as 3 May; the labels say 5 March.
    examples = [
        TrainingExample(
            facts=facts_from_label(
                label=make_label(date=f"2025-03-{day:02d}T10:00:00")
            ),
            evidence=make_evidence(
                rows=[(f"Datum {day:02d}-03-2025 10:00",), ("TOTAAL", "3,00")]
            ),
        )
        for day in range(1, 11)
    ]
    expert = DateExpert().fit(examples=examples)

    ranked = expert.suggest(
        evidence=make_evidence(rows=[("Datum 07-08-2025 12:00",)]), known={}
    )

    assert ranked[0][0] == date(2025, 8, 7)


def test_time_expert_returns_the_printed_time() -> None:
    expert = TimeExpert().fit(examples=_card_examples())
    receipt = make_evidence(
        rows=[("Datum 25-06-2025 10:11",), ("Open tot 22:00",)]
    )

    assert expert.suggest(evidence=receipt, known={})[0][0] == "10:11"


def test_class_expert_tells_categories_apart_by_receipt_text() -> None:
    expert = ClassExpert(field="category").fit(
        examples=_card_examples() + _cash_examples()
    )

    market = expert.suggest(
        evidence=cash_receipt(total="6,35", given="20,00", back="13,65"),
        known={},
    )

    assert market[0][0] == "food:market"
    assert (
        expert.suggest(evidence=card_receipt(total="6,35"), known={})[0][0]
        == "food:bakery"
    )


def test_class_expert_learns_the_payment_account_from_how_was_paid() -> None:
    expert = ClassExpert(field="bank_account").fit(
        examples=_card_examples() + _cash_examples()
    )

    assert (
        expert.suggest(evidence=card_receipt(total="6,35"), known={})[0][0]
        == "alex:examplebank:checking"
    )
    assert (
        expert.suggest(
            evidence=cash_receipt(total="6,35", given="10,00", back="3,65"),
            known={},
        )[0][0]
        == "alex:wallet:physical"
    )


def test_examples_without_a_label_for_the_field_are_left_out() -> None:
    # Withdrawals carry no category; they must not become a "None" class.
    examples = _card_examples()
    blank = TrainingExample(
        facts=facts_from_label(label=make_label(category=None)),
        evidence=card_receipt(total="1,00"),
    )
    expert = ClassExpert(field="category").fit(examples=examples + [blank])

    suggested = expert.suggest(evidence=card_receipt(total="6,35"), known={})

    assert [value for value, _ in suggested] == ["food:bakery"]


def _twin_shop_examples() -> List[TrainingExample]:
    """Two shops whose receipts read exactly the same."""
    examples = []
    for category, street, count in (
        ("food:bakery", "Bakkerstraat", 6),
        ("tools", "IJzerweg", 4),
    ):
        for _ in range(count):
            examples.append(
                TrainingExample(
                    facts=facts_from_label(
                        label=make_label(
                            category=category, shop_name="Winkel", street=street
                        )
                    ),
                    evidence=make_evidence(
                        rows=[("Winkel",), ("TOTAAL", "5,00")]
                    ),
                )
            )
    return examples


def test_class_expert_uses_a_confirmed_field_given_as_context() -> None:
    expert = ClassExpert(field="shop", context=("category",)).fit(
        examples=_twin_shop_examples()
    )
    receipt = make_evidence(rows=[("Winkel",), ("TOTAAL", "7,50")])

    blind = expert.suggest(evidence=receipt, known={})
    informed = expert.suggest(evidence=receipt, known={"category": "tools"})

    assert "Bakkerstraat" in blind[0][0]
    assert "IJzerweg" in informed[0][0]


def test_context_value_never_seen_before_falls_back_to_the_text() -> None:
    expert = ClassExpert(field="shop", context=("category",)).fit(
        examples=_twin_shop_examples()
    )
    receipt = make_evidence(rows=[("Winkel",), ("TOTAAL", "7,50")])

    informed = expert.suggest(evidence=receipt, known={"category": "travel"})

    assert "Bakkerstraat" in informed[0][0]


def test_unseen_subcategory_counts_as_its_parent_category() -> None:
    expert = ClassExpert(field="shop", context=("category",)).fit(
        examples=_twin_shop_examples()
    )
    receipt = make_evidence(rows=[("Winkel",), ("TOTAAL", "7,50")])

    informed = expert.suggest(
        evidence=receipt, known={"category": "tools:garden"}
    )

    assert "IJzerweg" in informed[0][0]


def test_known_fields_outside_the_context_are_ignored() -> None:
    expert = ClassExpert(field="shop", context=()).fit(
        examples=_twin_shop_examples()
    )
    receipt = make_evidence(rows=[("Winkel",), ("TOTAAL", "7,50")])

    assert expert.suggest(
        evidence=receipt, known={"category": "tools"}
    ) == expert.suggest(evidence=receipt, known={})


def test_having_a_linked_bank_transaction_at_all_is_informative() -> None:
    # Receipts linked to a bank transaction were paid by card, whatever
    # the other party was called; unlinked ones were paid in cash.
    def example(party, bank):
        return TrainingExample(
            facts=facts_from_label(
                label=make_label(
                    bank=bank, csv_other_party=party, shop_name="Winkel"
                )
            ),
            evidence=make_evidence(rows=[("Winkel",), ("TOTAAL", "5,00")]),
        )

    examples = [example(f"PARTIJ {n}", "examplebank") for n in range(6)]
    examples += [example(None, "wallet") for _ in range(8)]
    expert = ClassExpert(
        field="bank_account", context=("csv_transaction",)
    ).fit(examples=examples)
    receipt = make_evidence(rows=[("Winkel",), ("TOTAAL", "7,50")])

    linked = expert.suggest(
        evidence=receipt, known={"csv_transaction": "NOOIT GEZIENE PARTIJ"}
    )
    blind = expert.suggest(evidence=receipt, known={})

    assert linked[0][0] == "alex:examplebank:checking"
    assert blind[0][0] == "alex:wallet:checking"
