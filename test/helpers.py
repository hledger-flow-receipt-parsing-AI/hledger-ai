"""Invented fixtures that mirror the real label structure completely.

No value here comes from real data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typeguard import typechecked


@typechecked
def make_label(
    *,
    date: str = "2025-03-17T14:32:00",
    category: Optional[str] = "food:bakery",
    shop_name: str = "Voorbeeld Markt",
    street: Optional[str] = "Dorpsstraat",
    house_nr: Optional[str] = "12",
    zipcode: Optional[str] = "1234 AB",
    city: Optional[str] = "Voorbeeldstad",
    country: Optional[str] = "Netherlands",
    holder: str = "alex",
    bank: str = "examplebank",
    account_type: str = "checking",
    currency: str = "EUR",
    amount: float = 11.54,
    change: float = 0.0,
    subtotal: Optional[float] = None,
    total_tax: Optional[float] = 0.95,
    image_paths: Optional[List[str]] = None,
    csv_other_party: Optional[str] = None,
    csv_description: Optional[str] = None,
) -> Dict[str, Any]:
    """A purchase label as written by the labelling TUI."""
    original: Optional[Dict[str, Any]] = None
    if csv_other_party is not None or csv_description is not None:
        original = {
            "the_date": date,
            "description": csv_description or "",
            "other_party_name": csv_other_party or "",
            "amount": amount - change,
            "transaction_code": "DEBIT",
        }
    account = {
        "account_holder": holder,
        "bank": bank,
        "account_type": account_type,
        "base_currency": currency,
    }
    return {
        "raw_img_filepaths": image_paths or ["/data/in/receipt_0.jpg"],
        "the_date": date,
        "shop_identifier": {
            "name": shop_name,
            "address": {
                "street": street,
                "house_nr": house_nr,
                "zipcode": zipcode,
                "city": city,
                "country": country,
            },
            "shop_account_nr": None,
        },
        "net_bought_items": {
            "quantity": 1,
            "description": category,
            "the_date": date,
            "account_transactions": [
                {
                    "account": account,
                    "tendered_amount_out": amount,
                    "change_returned": change,
                    "currency": currency,
                    "the_date": date,
                    "original_transaction": original,
                    "parent_receipt_category": None,
                }
            ],
            "tax_per_unit": 0,
            "group_discount": 0,
            "category": None,
            "round_amount": None,
            "unit_price": None,
        },
        "net_returned_items": None,
        "subtotal": subtotal,
        "total_tax": total_tax,
        "receipt_owner_address": None,
        "ai_receipt_categorisation": None,
        "transaction_hash": None,
        "receipt_category": category,
        "withdrawal_metadata": None,
    }


@typechecked
def make_withdrawal_label(
    *, date: str = "2025-04-02T09:15:00", amount: float = 50.0
) -> Dict[str, Any]:
    """A cash withdrawal label: no category, items in net_returned_items."""
    label = make_label(date=date, category=None, amount=0.0, change=amount)
    label["net_returned_items"] = label["net_bought_items"]
    label["net_returned_items"]["account_transactions"][0]["account"] = {
        "account_holder": "alex",
        "bank": "wallet",
        "account_type": "physical",
        "base_currency": "EUR",
    }
    label["net_bought_items"] = None
    label["withdrawal_metadata"] = {
        "source_account_transaction": {
            "account": {
                "account_holder": "alex",
                "bank": "examplebank",
                "account_type": "checking",
                "base_currency": "EUR",
            },
            "tendered_amount_out": amount,
            "change_returned": 0.0,
            "currency": "EUR",
            "the_date": date,
            "original_transaction": None,
            "parent_receipt_category": None,
        },
        "atm_operator_fee": 0.0,
        "withdrawn_amount": None,
        "exchange_rate": None,
        "bank_fx_fee": 0.0,
    }
    return label


@typechecked
def make_evidence(
    *, rows: List[tuple], captured_at: Optional[Any] = None
) -> Any:
    """OCR evidence for an invented receipt: each row is a tuple of the text
    boxes printed on it, left to right."""
    from hledger_ai.ocr.types import OcrLine, OcrResult
    from hledger_ai.suggest.evidence import ReceiptEvidence

    lines = []
    for row_index, boxes in enumerate(rows):
        y = 40.0 + 50.0 * row_index
        for box_index, text in enumerate(boxes):
            x = 40.0 + 300.0 * box_index
            lines.append(
                OcrLine(
                    text=text,
                    score=0.99,
                    box=((x, y), (x + 200, y), (x + 200, y + 30), (x, y + 30)),
                )
            )
    result = OcrResult(
        engine="fake",
        image_sha256="0" * 64,
        width=700,
        height=int(80 + 50 * len(rows)),
        elapsed_s=0.0,
        lines=tuple(lines),
    )
    return ReceiptEvidence(ocr=(result,), captured_at=captured_at)


@typechecked
def card_receipt(
    *,
    total: str,
    shop: str = "Voorbeeld Markt",
    printed_date: str = "17-03-2025 14:32",
) -> Any:
    """A card payment: the total is repeated on the PIN row; a loyalty
    balance on the last row is larger than the total."""
    return make_evidence(
        rows=[
            (shop,),
            ("Dorpsstraat 12",),
            (f"Datum {printed_date}",),
            ("Brood", "2,49"),
            ("TOTAAL", total),
            ("PIN", total),
            ("BTW 9%", "0,95"),
            ("Spaarsaldo", "99,99"),
            ("Ruilen tot 31-03-2025",),
            ("Open tot 22:00",),
        ]
    )


@typechecked
def cash_receipt(
    *,
    total: str,
    given: str,
    back: str,
    printed_date: str = "18-03-2025 09:05",
) -> Any:
    return make_evidence(
        rows=[
            ("Markt De Kraam",),
            (f"Datum {printed_date}",),
            ("Kaas", total),
            ("TOTAAL", total),
            ("CONTANT", given),
            ("RETOUR", back),
        ]
    )
