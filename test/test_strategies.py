"""The cross-inference strategy registry (CI-5, ST-*)."""

import pytest

from hledger_ai.receipt_fields import FIELDS
from hledger_ai.strategies import (
    EVIDENCE_FIELDS,
    STRATEGIES,
    get_strategy,
    usable_strategies,
)


def test_image_alone_offers_only_first_guess_strategies_for_shop() -> None:
    ids = [s.id for s in usable_strategies(known={"image"}, improves="shop")]

    # ST-19 is the fallback that needs nothing: the most frequent shop.
    assert ids == ["ST-12", "ST-19"]


def test_confirming_the_category_unlocks_category_to_shop() -> None:
    ids = [
        s.id
        for s in usable_strategies(known={"image", "category"}, improves="shop")
    ]

    assert ids == ["ST-1", "ST-12", "ST-19"]


def test_strategy_needing_two_known_fields_waits_for_both() -> None:
    only_date = usable_strategies(known={"date"}, improves="csv_transaction")
    both = usable_strategies(
        known={"date", "amount", "change"}, improves="csv_transaction"
    )

    assert [s.id for s in only_date] == []
    assert [s.id for s in both] == ["ST-5"]


def test_strategies_without_usable_labels_are_never_offered() -> None:
    # Item descriptions mirror the category in today's labels, so ST-7 and
    # ST-8 have nothing to learn from.
    offered = usable_strategies(
        known={"image", "shop", "item_description"}, improves="category"
    )

    assert "ST-8" not in [s.id for s in offered]
    assert get_strategy(strategy_id="ST-8").applicable is False


def test_unknown_strategy_id_is_an_error() -> None:
    with pytest.raises(KeyError):
        get_strategy(strategy_id="ST-999")


def test_every_strategy_refers_to_existing_fields_and_ids_are_unique() -> None:
    valid = set(FIELDS) | set(EVIDENCE_FIELDS)
    ids = [s.id for s in STRATEGIES]

    assert len(ids) == len(set(ids))
    for strategy in STRATEGIES:
        assert set(strategy.known) <= valid, strategy.id
        assert set(strategy.improves) <= valid, strategy.id
        assert strategy.tier in (1, 2, 3), strategy.id
