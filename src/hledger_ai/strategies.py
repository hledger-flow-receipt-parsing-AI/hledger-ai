"""Registry of cross-inference strategies (requirements CI-5, ST-*).

A strategy says: when these fields are known, that field can be improved.
IDs are stable: never renumber, retire an ID instead of reusing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

from typeguard import typechecked

# Things a strategy can use or improve that are not labeller questions.
EVIDENCE_FIELDS: Tuple[str, ...] = (
    "image",  # the receipt photo (through an OCR engine)
    "csv_transaction",  # the bank CSV transaction linked to the receipt
    "shop_name",  # manual address entry: the name typed first
    "address",  # manual address entry: street, zip code or city
    "item_description",
)


@typechecked
@dataclass(frozen=True)
class Strategy:
    """One way to improve fields from fields that are already known."""

    id: str
    known: Tuple[str, ...]
    improves: Tuple[str, ...]
    tier: int  # cheapest training tier expected to work (TR-1)
    applicable: bool = True
    note: str = ""


STRATEGIES: Tuple[Strategy, ...] = (
    Strategy(id="ST-1", known=("category",), improves=("shop",), tier=1),
    Strategy(id="ST-2", known=("shop",), improves=("category",), tier=1),
    Strategy(id="ST-3", known=("shop_name",), improves=("shop",), tier=1),
    Strategy(id="ST-4", known=("address",), improves=("shop",), tier=1),
    Strategy(
        id="ST-5",
        known=("date", "amount", "change"),
        improves=("csv_transaction",),
        tier=1,
    ),
    Strategy(
        id="ST-6",
        known=("csv_transaction",),
        improves=("shop", "category", "bank_account", "date", "amount"),
        tier=1,
    ),
    Strategy(
        id="ST-7",
        known=("shop",),
        improves=("item_description",),
        tier=1,
        applicable=False,
        note="item descriptions mirror the category in today's labels",
    ),
    Strategy(
        id="ST-8",
        known=("item_description",),
        improves=("category",),
        tier=1,
        applicable=False,
        note="item descriptions mirror the category in today's labels",
    ),
    Strategy(
        id="ST-9",
        known=("item_description", "total_tax"),
        improves=("subtotal", "amount"),
        tier=1,
        applicable=False,
        note="labels hold no separate items to add up",
    ),
    # First guesses from the image alone (CI-2).
    Strategy(id="ST-10", known=("image",), improves=("date", "time"), tier=2),
    Strategy(
        id="ST-11", known=("image",), improves=("amount", "change"), tier=2
    ),
    Strategy(id="ST-12", known=("image",), improves=("shop",), tier=2),
    Strategy(
        id="ST-13",
        known=("image",),
        improves=("category", "is_withdrawal"),
        tier=2,
    ),
    Strategy(
        id="ST-14",
        known=("image",),
        improves=("bank_account", "currency"),
        tier=2,
    ),
    Strategy(
        id="ST-15",
        known=("image",),
        improves=("subtotal", "total_tax"),
        tier=2,
    ),
    Strategy(
        id="ST-16", known=("category",), improves=("bank_account",), tier=1
    ),
    Strategy(
        id="ST-17",
        known=("bank_account",),
        improves=("currency", "change"),
        tier=1,
    ),
    Strategy(
        id="ST-18", known=("image", "amount"), improves=("change",), tier=2
    ),
    Strategy(
        id="ST-20",
        known=("image", "amount", "change"),
        improves=("subtotal", "total_tax"),
        tier=2,
    ),
    # Nothing known at all: the value that is usually right. This is also
    # the baseline the image strategies have to beat (EV-3).
    Strategy(
        id="ST-19",
        known=(),
        improves=(
            "is_withdrawal",
            "category",
            "bank_account",
            "currency",
            "change",
            "shop",
        ),
        tier=1,
    ),
)


@typechecked
def get_strategy(*, strategy_id: str) -> Strategy:
    for strategy in STRATEGIES:
        if strategy.id == strategy_id:
            return strategy
    raise KeyError(f"Unknown strategy: {strategy_id}")


@typechecked
def usable_strategies(*, known: Set[str], improves: str) -> List[Strategy]:
    """Applicable strategies that improve *improves* from *known* alone."""
    return [
        strategy
        for strategy in STRATEGIES
        if strategy.applicable
        and improves in strategy.improves
        and set(strategy.known) <= known
    ]
