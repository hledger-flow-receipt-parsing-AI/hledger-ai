"""Evaluation: time-ordered folds, baseline and per-strategy accuracy (EV-*)."""

from typing import List

from helpers import card_receipt, cash_receipt, make_label

from hledger_ai.evaluation.folds import forward_chaining_folds
from hledger_ai.evaluation.scoring import (
    Baseline,
    comparable,
    evaluate_fold,
)
from hledger_ai.models.experts import TrainingExample
from hledger_ai.receipt_fields import facts_from_label
from hledger_ai.strategies import get_strategy


def test_folds_train_on_the_past_and_test_on_the_next_chunk() -> None:
    folds = forward_chaining_folds(count=10, folds=4)

    assert [(list(train), list(test)) for train, test in folds] == [
        ([0, 1], [2, 3]),
        ([0, 1, 2, 3], [4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
    ]


def test_single_fold_is_one_split_with_the_newest_receipts_held_out() -> None:
    folds = forward_chaining_folds(count=10, folds=1)

    assert [(list(train), list(test)) for train, test in folds] == [
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    ]


def test_too_few_receipts_give_fewer_folds_never_empty_ones() -> None:
    folds = forward_chaining_folds(count=3, folds=4)

    assert [(list(train), list(test)) for train, test in folds] == [
        ([0], [1]),
        ([0, 1], [2]),
    ]
    assert forward_chaining_folds(count=1, folds=4) == []


def test_values_compare_by_meaning_not_by_type() -> None:
    assert comparable(field="amount", value=11.54) == comparable(
        field="amount", value=11.540000001
    )
    assert comparable(field="is_withdrawal", value=True) == comparable(
        field="is_withdrawal", value="y"
    )
    assert comparable(field="amount", value=None) is None


def _example(*, category: str, shop: str, bank: str) -> TrainingExample:
    return TrainingExample(
        facts=facts_from_label(
            label=make_label(category=category, shop_name=shop, bank=bank)
        ),
        evidence=card_receipt(total="1,00", shop=shop),
    )


def test_baseline_is_the_most_frequent_value_given_the_known_fields() -> None:
    train = (
        [_example(category="food", shop="Bakker Bart", bank="b1")] * 3
        + [_example(category="food", shop="Broodhuis", bank="b1")]
        + [_example(category="tools", shop="IJzerwaren", bank="b2")] * 2
    )
    baseline = Baseline.fit(examples=train)
    tools = facts_from_label(
        label=make_label(category="tools", shop_name="Onbekend", bank="b2")
    )

    given_category = baseline.predict(
        strategy=get_strategy(strategy_id="ST-1"), field="shop", facts=tools
    )
    blind = baseline.predict(
        strategy=get_strategy(strategy_id="ST-12"), field="shop", facts=tools
    )

    assert given_category.startswith("IJzerwaren: ")
    assert blind.startswith("Bakker Bart: ")


def _history() -> List[TrainingExample]:
    examples = []
    for total in ("11,54", "7,20", "23,05", "4,99", "15,75", "31,10", "8,40"):
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
                        shop_name="Markt De Kraam",
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


def test_image_model_is_scored_per_strategy_and_field_against_baseline() -> (
    None
):
    history = _history()
    train, test = history[:10], history[10:]

    scores = evaluate_fold(train=train, test=test, use_image=True)

    amount = scores["ST-11"]["amount"]
    assert (amount.correct, amount.total) == (4, 4)
    # The most frequent amount so far (50,00) is right for the two cash
    # receipts only.
    assert amount.baseline_correct == 2
    category = scores["ST-13"]["category"]
    assert (category.correct, category.total) == (4, 4)
    assert category.baseline_correct == 2


def test_priors_model_has_no_score_for_image_strategies() -> None:
    history = _history()

    scores = evaluate_fold(
        train=history[:10], test=history[10:], use_image=False
    )

    assert "ST-11" not in scores
    account = scores["ST-16"]["bank_account"]
    assert (account.correct, account.total) == (4, 4)
    # A prior is itself "the most frequent value given the known field".
    assert account.baseline_correct == 4


def test_strategies_whose_known_field_is_unlabelled_are_left_out() -> None:
    # No receipt in this history is linked to a bank transaction.
    history = _history()

    scores = evaluate_fold(
        train=history[:10], test=history[10:], use_image=True
    )

    assert "ST-6" not in scores


def test_scores_only_count_strategies_asked_for() -> None:
    history = _history()

    scores = evaluate_fold(
        train=history[:10],
        test=history[10:],
        use_image=True,
        strategy_ids={"ST-10"},
    )

    assert list(scores) == ["ST-10"]
    assert scores["ST-10"]["date"].total == 4


def test_the_nothing_known_strategy_is_scored_for_priors_only() -> None:
    history = _history()

    with_image = evaluate_fold(
        train=history[:10], test=history[10:], use_image=True
    )
    priors = evaluate_fold(
        train=history[:10], test=history[10:], use_image=False
    )

    assert "ST-19" not in with_image
    assert priors["ST-19"]["currency"].correct == 4
