"""Where on the photo the evidence for a suggestion is (boxes to draw).

Amounts, dates and times point at the OCR boxes that hold the suggested
value; class fields (payment account, category, ...) point at the boxes
whose text pushed the model towards its answer. All receipts are invented.
"""

from datetime import date
from typing import List

from helpers import card_receipt, cash_receipt, make_evidence, make_label
from test_models_experts import TOTALS, _card_examples, _cash_examples

from hledger_ai.highlights import (
    Highlight,
    amount_highlights,
    date_highlights,
    evidence_highlights,
    field_highlights,
    located_rows,
    time_highlights,
)
from hledger_ai.models.experts import ClassExpert, TrainingExample
from hledger_ai.receipt_fields import facts_from_label
from hledger_ai.suggest.engine import Suggestion, SuggestionEngine
from hledger_ai.suggest.evidence import ReceiptEvidence

PHOTO = "/data/proc/a_cropped.jpg"
SHOPS = [
    "Bakkerij Noord",
    "Slagerij West",
    "Groentehal",
    "Kaashuis",
    "Boekhandel",
    "Fietsenmaker",
    "Drogist Zuid",
    "Bloemist Oost",
]
STALLS = [
    "Vismarkt",
    "Fruitstal",
    "Kaasboer",
    "Notenkraam",
    "Olijvenbar",
    "Honingstal",
    "Broodkraam",
    "Kruidenstal",
]


def _payment_examples() -> List[TrainingExample]:
    """Card and cash receipts from different shops each: only the way of
    paying (PIN or CONTANT/RETOUR) is the same within a kind."""
    examples = []
    for index, total in enumerate(TOTALS):
        paid = float(total.replace(",", "."))
        examples.append(
            TrainingExample(
                facts=facts_from_label(
                    label=make_label(
                        amount=paid, change=0.0, shop_name=SHOPS[index]
                    )
                ),
                evidence=make_evidence(
                    rows=[
                        (SHOPS[index],),
                        ("Datum 17-03-2025 14:32",),
                        ("Artikel", total),
                        ("TOTAAL", total),
                        ("PIN", total),
                    ]
                ),
            )
        )
        back = f"{50 - paid:.2f}".replace(".", ",")
        examples.append(
            TrainingExample(
                facts=facts_from_label(
                    label=make_label(
                        amount=50.0,
                        change=round(50 - paid, 2),
                        bank="wallet",
                        account_type="physical",
                        shop_name=STALLS[index],
                    )
                ),
                evidence=make_evidence(
                    rows=[
                        (STALLS[index],),
                        ("Datum 17-03-2025 14:32",),
                        ("Artikel", total),
                        ("TOTAAL", total),
                        ("CONTANT", "50,00"),
                        ("RETOUR", back),
                    ]
                ),
            )
        )
    return examples


def _box(*, row: int, column: int = 0):
    """Where make_evidence puts the box of *column* on *row*."""
    x, y = 40.0 + 300.0 * column, 40.0 + 50.0 * row
    return ((x, y), (x + 200, y), (x + 200, y + 30), (x, y + 30))


def _rows(evidence=None):
    return located_rows(
        evidence=evidence or card_receipt(total="6,35"), image_paths=[PHOTO]
    )


def test_amount_highlights_are_the_boxes_printing_that_amount() -> None:
    found = amount_highlights(rows=_rows(), value=6.35)

    assert found == [
        Highlight(image_path=PHOTO, box=_box(row=4, column=1), text="6,35"),
        Highlight(image_path=PHOTO, box=_box(row=5, column=1), text="6,35"),
    ]


def test_an_amount_that_is_not_printed_has_no_highlight() -> None:
    assert amount_highlights(rows=_rows(), value=99.0) == []


def test_date_and_time_highlights_are_the_box_printing_them() -> None:
    assert date_highlights(rows=_rows(), value=date(2025, 3, 17)) == [
        Highlight(
            image_path=PHOTO, box=_box(row=2), text="Datum 17-03-2025 14:32"
        )
    ]
    assert time_highlights(rows=_rows(), value="22:00") == [
        Highlight(image_path=PHOTO, box=_box(row=9), text="Open tot 22:00")
    ]


def test_a_date_split_over_boxes_highlights_its_whole_row() -> None:
    rows = _rows(make_evidence(rows=[("17", "maart 2025"), ("Kaas",)]))

    assert [
        h.box for h in date_highlights(rows=rows, value=date(2025, 3, 17))
    ] == [
        _box(row=0, column=0),
        _box(row=0, column=1),
    ]


def test_rows_of_a_second_photo_carry_that_photo_path() -> None:
    first = make_evidence(rows=[("TOTAAL", "6,35")])
    second = make_evidence(rows=[("PIN", "6,35")])
    evidence = ReceiptEvidence(ocr=(first.ocr[0], second.ocr[0]))

    found = amount_highlights(
        rows=located_rows(evidence=evidence, image_paths=[PHOTO, "/b.jpg"]),
        value=6.35,
    )

    assert [h.image_path for h in found] == [PHOTO, "/b.jpg"]
    assert [h.box for h in found] == [_box(row=0, column=1)] * 2


def test_evidence_highlights_are_the_boxes_that_point_to_the_class() -> None:
    expert = ClassExpert(field="bank_account").fit(
        examples=_card_examples() + _cash_examples()
    )
    rows = _rows(cash_receipt(total="6,35", given="50,00", back="43,65"))

    found = evidence_highlights(
        rows=rows,
        scores=lambda texts: expert.text_scores(
            texts=texts, value="alex:wallet:physical"
        ),
    )

    assert "CONTANT" in [h.text for h in found]
    assert "6,35" not in [h.text for h in found]
    assert len(found) <= 3


def test_no_evidence_when_nothing_points_to_the_class() -> None:
    assert (
        evidence_highlights(
            rows=_rows(), scores=lambda texts: [0.0] * len(texts)
        )
        == []
    )


def test_field_highlights_follow_the_suggested_values() -> None:
    engine = SuggestionEngine.train(
        examples=_card_examples() + _cash_examples()
    )
    evidence = cash_receipt(total="6,35", given="50,00", back="43,65")

    found = field_highlights(
        engine=engine,
        evidence=evidence,
        image_paths=[PHOTO],
        suggestions={
            "amount": [Suggestion(50.0, 0.9, ("ST-11",))],
            "change": [Suggestion(43.65, 0.8, ("ST-11",))],
            "date": [Suggestion(date(2025, 3, 18), 0.9, ("ST-10",))],
            "time": [Suggestion("09:05", 0.9, ("ST-10",))],
            "bank_account": [
                Suggestion("alex:wallet:physical", 0.9, ("ST-14",))
            ],
            "currency": [Suggestion("EUR", 0.99, ("ST-19",))],
        },
    )

    assert [h.text for h in found["amount"]] == ["50,00"]
    assert [h.text for h in found["change"]] == ["43,65"]
    assert [h.text for h in found["date"]] == ["Datum 18-03-2025 09:05"]
    assert found["time"] == found["date"]
    assert "CONTANT" in [h.text for h in found["bank_account"]]
    # One currency in every label: nothing to tell apart, nothing to show.
    assert "currency" not in found


def test_no_change_is_explained_by_the_payment_boxes() -> None:
    engine = SuggestionEngine.train(examples=_payment_examples())
    receipt = make_evidence(
        rows=[
            ("Speelgoedwinkel",),
            ("Datum 02-04-2025 11:10",),
            ("Artikel", "6,35"),
            ("TOTAAL", "6,35"),
            ("PIN", "6,35"),
        ]
    )

    found = field_highlights(
        engine=engine,
        evidence=receipt,
        image_paths=[PHOTO],
        suggestions={"change": [Suggestion(0.0, 0.9, ("ST-11",))]},
    )

    assert [h.text for h in found["change"]] == ["PIN"]
