"""The receipt label flattened into the fields the labeller asks for.

Field ids are shared by strategies, models, evaluation and the labeller
adapter. Labels are read as plain dicts (the JSON the labeller writes), so
this module never touches or rewrites a label file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from hledger_core.TransactionObjects.Address import Address
from typeguard import typechecked

# Every field a model can suggest, in the order the labeller asks for them.
FIELDS: Tuple[str, ...] = (
    "date",
    "time",
    "is_withdrawal",
    "category",
    "bank_account",
    "currency",
    "amount",
    "change",
    "shop",
    "subtotal",
    "total_tax",
)

# Bump when the label structure read below changes (DS-1).
LABEL_SCHEMA_VERSION = "1"


# Amounts a receipt may lack: leaving the question empty is an answer too
# (most receipts print no subtotal), which the models learn and suggest as
# the value None.
OPTIONAL_AMOUNTS: Tuple[str, ...] = ("subtotal",)


@typechecked
@dataclass(frozen=True)
class ReceiptFacts:
    """One labelled receipt as flat field values.

    *complete* is False for a partial label (a receipt moved away as "not
    mine" halfway): a field missing there was not answered, so it says
    nothing about whether the receipt has it.
    """

    the_datetime: Optional[datetime]
    category: Optional[str]
    shop_name: Optional[str]
    street: Optional[str]
    house_nr: Optional[str]
    zipcode: Optional[str]
    city: Optional[str]
    country: Optional[str]
    bank_account: Optional[str]
    currency: Optional[str]
    amount: Optional[float]
    change: Optional[float]
    subtotal: Optional[float]
    total_tax: Optional[float]
    is_withdrawal: bool
    csv_other_party: Optional[str]
    csv_description: Optional[str]
    image_paths: Tuple[str, ...]
    complete: bool = True

    def shop(self) -> Optional[str]:
        """The shop exactly as the labeller's address list shows it."""
        if not self.shop_name:
            return None
        return shop_choice_text(
            name=self.shop_name,
            street=self.street,
            house_nr=self.house_nr,
            zipcode=self.zipcode,
            city=self.city,
            country=self.country,
        )

    def get(self, *, field: str) -> Any:
        """Value of *field* (one of ``FIELDS``), or None when unlabelled."""
        if field == "date":
            return self.the_datetime.date() if self.the_datetime else None
        if field == "time":
            return (
                self.the_datetime.strftime("%H:%M")
                if self.the_datetime
                else None
            )
        if field == "shop":
            return self.shop()
        if field not in FIELDS:
            raise KeyError(f"Unknown receipt field: {field}")
        return getattr(self, field)


@typechecked
def shop_choice_text(
    *,
    name: str,
    street: Optional[str],
    house_nr: Optional[str],
    zipcode: Optional[str],
    city: Optional[str],
    country: Optional[str],
) -> str:
    """Same text as tui-image-labeller's address selector choices."""
    address = Address(
        street=street,
        house_nr=house_nr,
        zipcode=zipcode,
        city=city,
        country=country,
    )
    return f"{name}: {address.to_string()}"


@typechecked
def _parse_datetime(*, text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@typechecked
def _as_float(*, value: Any) -> Optional[float]:
    return None if value is None else float(value)


@typechecked
def facts_from_label(*, label: Dict[str, Any]) -> ReceiptFacts:
    """Flatten one label dict (``receipt_image_to_obj_label.json``)."""
    shop = label.get("shop_identifier") or {}
    address = shop.get("address") or {}
    item = label.get("net_bought_items") or label.get("net_returned_items")
    transactions = (item or {}).get("account_transactions") or []
    first = transactions[0] if transactions else {}
    account = first.get("account") or {}
    original = first.get("original_transaction") or {}

    paths = label.get("raw_img_filepaths", label.get("raw_img_filepath"))
    if isinstance(paths, str):
        paths = [paths]

    bank_account: Optional[str] = None
    if account:
        bank_account = (
            f"{account.get('account_holder')}:{account.get('bank')}"
            f":{account.get('account_type')}"
        )
    return ReceiptFacts(
        the_datetime=_parse_datetime(text=label.get("the_date")),
        category=label.get("receipt_category"),
        shop_name=shop.get("name"),
        street=address.get("street"),
        house_nr=address.get("house_nr"),
        zipcode=address.get("zipcode"),
        city=address.get("city"),
        country=address.get("country"),
        bank_account=bank_account,
        currency=first.get("currency") or account.get("base_currency"),
        amount=_as_float(value=first.get("tendered_amount_out")),
        change=_as_float(value=first.get("change_returned")),
        subtotal=_as_float(value=label.get("subtotal")),
        total_tax=_as_float(value=label.get("total_tax")),
        is_withdrawal=label.get("withdrawal_metadata") is not None,
        csv_other_party=original.get("other_party_name") or None,
        csv_description=original.get("description") or None,
        image_paths=tuple(paths or ()),
    )


@typechecked
def known_key(*, known_field: str, value: Any) -> Optional[str]:
    """A confirmed value as the text that priors and models are keyed on."""
    if value is None or value == "":
        return None
    text = str(value)
    if known_field == "address":
        return re.sub(r"\s+", "", text).upper()
    if known_field in ("csv_transaction", "shop_name"):
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    return text


@typechecked
def known_from_facts(*, facts: ReceiptFacts) -> Dict[str, Any]:
    """Everything a user could have confirmed, taken from a label."""
    known: Dict[str, Any] = {field: facts.get(field=field) for field in FIELDS}
    known["shop_name"] = facts.shop_name
    known["address"] = facts.zipcode
    known["csv_transaction"] = facts.csv_other_party
    return {k: v for k, v in known.items() if v is not None}


__all__ = [
    "FIELDS",
    "LABEL_SCHEMA_VERSION",
    "ReceiptFacts",
    "facts_from_label",
    "known_from_facts",
    "known_key",
    "shop_choice_text",
]
