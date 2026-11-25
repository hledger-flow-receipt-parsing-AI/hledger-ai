"""Flattening a label dict into the fields the labeller asks for."""

from datetime import date

from helpers import make_label, make_withdrawal_label

from hledger_ai.receipt_fields import facts_from_label


def test_purchase_label_yields_every_labeller_field() -> None:
    facts = facts_from_label(
        label=make_label(amount=20.0, change=8.46, subtotal=10.59)
    )

    assert facts.get(field="date") == date(2025, 3, 17)
    assert facts.get(field="time") == "14:32"
    assert facts.get(field="category") == "food:bakery"
    assert facts.get(field="bank_account") == "alex:examplebank:checking"
    assert facts.get(field="currency") == "EUR"
    assert facts.get(field="amount") == 20.0
    assert facts.get(field="change") == 8.46
    assert facts.get(field="subtotal") == 10.59
    assert facts.get(field="total_tax") == 0.95
    assert facts.get(field="is_withdrawal") is False


def test_shop_value_equals_the_labellers_address_choice_text() -> None:
    # The labeller lists shops as "<name>: <street>, <nr>, <zip>, <city>,
    # <country>"; a different text could not be matched to a choice.
    facts = facts_from_label(label=make_label())

    assert facts.get(field="shop") == (
        "Voorbeeld Markt: Dorpsstraat, 12, 1234 AB, Voorbeeldstad,"
        " Netherlands"
    )


def test_shop_text_skips_missing_address_parts() -> None:
    facts = facts_from_label(
        label=make_label(street=None, house_nr=None, zipcode=None)
    )

    assert facts.get(field="shop") == (
        "Voorbeeld Markt: Voorbeeldstad, Netherlands"
    )


def test_withdrawal_label_is_flagged_and_has_no_category() -> None:
    facts = facts_from_label(label=make_withdrawal_label())

    assert facts.get(field="is_withdrawal") is True
    assert facts.get(field="category") is None
    assert facts.get(field="bank_account") == "alex:wallet:physical"


def test_legacy_single_image_path_is_read() -> None:
    label = make_label()
    del label["raw_img_filepaths"]
    label["raw_img_filepath"] = "/data/in/old.jpg"

    assert facts_from_label(label=label).image_paths == ("/data/in/old.jpg",)


def test_linked_bank_transaction_text_is_exposed() -> None:
    facts = facts_from_label(
        label=make_label(
            csv_other_party="VOORBEELD MARKT BV", csv_description="PAS 123"
        )
    )

    assert facts.csv_other_party == "VOORBEELD MARKT BV"
    assert facts.csv_description == "PAS 123"
