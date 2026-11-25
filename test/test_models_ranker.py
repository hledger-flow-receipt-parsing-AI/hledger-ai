"""Tier-2 candidate ranker: learns which candidate on a receipt is right."""

from typing import Dict, List

import pytest

from hledger_ai.models.ranker import CandidateRanker, rank_values


def _receipt(total_row_word: str) -> List[Dict[str, float]]:
    """Three amounts; only the middle one sits on the row with the word."""
    return [
        {"position": 0.2, "w:brood": 1.0},
        {"position": 0.8, f"w:{total_row_word}": 1.0},
        {"position": 0.9, "w:btw": 1.0},
    ]


TRAIN_GROUPS = [_receipt("totaal") for _ in range(12)]
TRAIN_CORRECT = [[False, True, False] for _ in range(12)]


def test_ranker_prefers_the_candidate_that_looks_like_past_answers() -> None:
    ranker = CandidateRanker().fit(groups=TRAIN_GROUPS, correct=TRAIN_CORRECT)

    probabilities = ranker.probabilities(features=_receipt("totaal"))

    assert probabilities.index(max(probabilities)) == 1
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_unseen_feature_names_are_ignored_not_fatal() -> None:
    ranker = CandidateRanker().fit(groups=TRAIN_GROUPS, correct=TRAIN_CORRECT)

    probabilities = ranker.probabilities(
        features=[{"position": 0.8, "w:totaal": 1.0, "w:nooitgezien": 1.0}]
    )

    assert probabilities[0] > 0.5


def test_ranker_without_any_positive_example_stays_untrained() -> None:
    ranker = CandidateRanker().fit(
        groups=[_receipt("totaal")], correct=[[False, False, False]]
    )

    assert ranker.is_trained is False
    assert ranker.probabilities(features=_receipt("totaal")) == [0.0, 0.0, 0.0]


def test_ranker_that_only_ever_saw_right_candidates_trusts_any_candidate() -> (
    None
):
    # E.g. every training receipt printed exactly one time, the right one.
    ranker = CandidateRanker().fit(
        groups=[[{"position": 0.1}], [{"position": 0.2}]],
        correct=[[True], [True]],
    )

    assert ranker.probabilities(features=[{"position": 0.5}]) == [1.0]


def test_ranker_with_no_candidates_returns_nothing() -> None:
    ranker = CandidateRanker().fit(groups=TRAIN_GROUPS, correct=TRAIN_CORRECT)

    assert ranker.probabilities(features=[]) == []


def test_same_value_printed_twice_counts_as_stronger_evidence() -> None:
    ranked = rank_values(
        values=[11.54, 20.0, 11.54], probabilities=[0.5, 0.6, 0.5]
    )

    # 1 - (1 - 0.5) * (1 - 0.5) = 0.75 beats the single 0.6.
    assert [value for value, _ in ranked] == [11.54, 20.0]
    assert ranked[0][1] == pytest.approx(0.75 / 1.35)
    assert ranked[1][1] == pytest.approx(0.6 / 1.35)


def test_confidences_that_already_fit_under_one_are_left_alone() -> None:
    ranked = rank_values(values=["a", "b"], probabilities=[0.6, 0.1])

    assert ranked == [("a", pytest.approx(0.6)), ("b", pytest.approx(0.1))]
