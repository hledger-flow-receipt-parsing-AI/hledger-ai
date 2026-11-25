"""Tier-1 priors: what usually goes with an already known field."""

import math

import pytest

from hledger_ai.models.priors import FrequencyPrior

PAIRS = (
    [("food:bakery", "Bakker Bart")] * 3
    + [("food:bakery", "Broodhuis")]
    + [("tools", "IJzerwaren Jansen")] * 5
)


def test_known_value_ranks_the_targets_seen_with_it_first() -> None:
    prior = FrequencyPrior.fit(pairs=PAIRS)

    ranked = [target for target, _ in prior.ranked(known="food:bakery")]

    assert ranked == ["Bakker Bart", "Broodhuis", "IJzerwaren Jansen"]


def test_probabilities_add_up_to_one() -> None:
    prior = FrequencyPrior.fit(pairs=PAIRS)

    total = sum(p for _, p in prior.ranked(known="food:bakery"))

    assert total == pytest.approx(1.0)


def test_unseen_known_value_falls_back_to_overall_frequency() -> None:
    prior = FrequencyPrior.fit(pairs=PAIRS)

    ranked = [target for target, _ in prior.ranked(known="clothing")]

    assert ranked == ["IJzerwaren Jansen", "Bakker Bart", "Broodhuis"]


def test_unseen_subcategory_backs_off_to_its_parent_category() -> None:
    # Overall the hardware shop is most frequent, but a new kind of bakery
    # purchase should still suggest the bakeries.
    prior = FrequencyPrior.fit(pairs=PAIRS, hierarchical=True)

    ranked = [target for target, _ in prior.ranked(known="food:bakery:cake")]

    assert ranked[0] == "Bakker Bart"


def test_without_hierarchy_a_colon_in_the_value_means_nothing() -> None:
    prior = FrequencyPrior.fit(pairs=PAIRS, hierarchical=False)

    ranked = [target for target, _ in prior.ranked(known="food:bakery:cake")]

    assert ranked[0] == "IJzerwaren Jansen"


def test_log_ratio_is_positive_for_supported_and_negative_for_others() -> None:
    prior = FrequencyPrior.fit(pairs=PAIRS)

    ratios = prior.log_ratios(known="food:bakery")

    assert ratios["Bakker Bart"] > 0
    assert ratios["IJzerwaren Jansen"] < 0
    assert all(math.isfinite(v) for v in ratios.values())


def test_prior_survives_saving_and_loading() -> None:
    prior = FrequencyPrior.fit(pairs=PAIRS, hierarchical=True)

    restored = FrequencyPrior.from_json(data=prior.to_json())

    assert restored.ranked(known="food:bakery:cake") == prior.ranked(
        known="food:bakery:cake"
    )


def test_prior_without_any_data_suggests_nothing() -> None:
    assert FrequencyPrior.fit(pairs=[]).ranked(known="food") == []
