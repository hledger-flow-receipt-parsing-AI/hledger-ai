"""Choosing the model and strategy for a field (EV-3, EV-4, SP-5, TR-1)."""

from typing import Dict, Optional, Tuple

from hledger_ai.suggest.policy import ModelView, choose
from hledger_ai.training.state import ModelState, StrategyState


def _strategy(
    field: str, accuracy: float, baseline: float, seconds: float = 0.01
) -> StrategyState:
    return StrategyState(
        infer_done=100,
        infer_total=100,
        infer_seconds=seconds * 100,
        folds={
            "0": {
                field: {
                    "correct": round(accuracy * 1000),
                    "total": 1000,
                    "baseline_correct": round(baseline * 1000),
                }
            }
        },
    )


def _model(
    name: str,
    tier: int,
    uses_image: bool,
    strategies: Dict[str, StrategyState],
    finished: bool = True,
) -> ModelView:
    state = ModelState(
        model=name,
        dataset_hash="abc",
        private=True,
        model_version="2",
        hyperparameters={},
        finished=finished,
        strategies=strategies,
    )
    return ModelView(name=name, tier=tier, uses_image=uses_image, state=state)


PRIORS = _model(
    "priors",
    1,
    False,
    {
        "ST-1": _strategy("shop", 0.60, 0.60),
        "ST-4": _strategy("shop", 0.64, 0.64),
        "ST-19": _strategy("shop", 0.29, 0.29),
    },
)
LINEAR = _model(
    "rapidocr-linear",
    2,
    True,
    {
        "ST-1": _strategy("shop", 0.66, 0.60),
        "ST-4": _strategy("shop", 0.63, 0.64),
        "ST-12": _strategy("shop", 0.63, 0.29),
    },
)


def _pick(
    field: str = "shop",
    known: Optional[set] = None,
    has_image: bool = True,
    models: Tuple[ModelView, ...] = (PRIORS, LINEAR),
    budget: float = 23.0,
) -> Optional[Tuple[str, str]]:
    choice = choose(
        field=field,
        known=known or set(),
        has_image=has_image,
        models=models,
        budget_seconds=lambda strategy_id: budget,
    )
    return (choice.model, choice.strategy_id) if choice else None


def test_first_guess_comes_from_the_image_model() -> None:
    assert _pick() == ("rapidocr-linear", "ST-12")


def test_confirmed_category_switches_to_the_more_accurate_strategy() -> None:
    assert _pick(known={"category"}) == ("rapidocr-linear", "ST-1")


def test_prior_wins_where_it_is_more_accurate_than_the_image_model() -> None:
    # Zip code -> shop: the tier-2 model does not beat the prior here.
    assert _pick(known={"address"}) == ("priors", "ST-4")


def test_without_a_photo_only_priors_can_answer() -> None:
    assert _pick(has_image=False) == ("priors", "ST-19")
    assert _pick(known={"category"}, has_image=False) == ("priors", "ST-1")


def test_model_that_does_not_beat_its_baseline_is_not_used() -> None:
    priors = _model(
        "priors", 1, False, {"ST-19": _strategy("currency", 0.99, 0.99)}
    )
    linear = _model(
        "rapidocr-linear", 2, True, {"ST-14": _strategy("currency", 0.99, 0.99)}
    )

    assert _pick(field="currency", models=(priors, linear)) == (
        "priors",
        "ST-19",
    )
    assert _pick(field="currency", models=(linear,)) is None


def test_known_field_that_makes_things_worse_is_ignored() -> None:
    linear = _model(
        "rapidocr-linear",
        2,
        True,
        {
            "ST-14": _strategy("bank_account", 0.86, 0.45),
            "ST-16": _strategy("bank_account", 0.83, 0.53),
        },
    )

    assert _pick(
        field="bank_account", known={"category"}, models=(linear,)
    ) == ("rapidocr-linear", "ST-14")


def test_strategy_slower_than_its_time_budget_is_skipped() -> None:
    slow = _model(
        "slow-model",
        3,
        True,
        {"ST-12": _strategy("shop", 0.90, 0.29, seconds=40.0)},
    )

    assert _pick(models=(PRIORS, LINEAR, slow)) == ("rapidocr-linear", "ST-12")
    assert _pick(models=(PRIORS, LINEAR, slow), budget=60.0) == (
        "slow-model",
        "ST-12",
    )


def test_equal_accuracy_goes_to_the_cheaper_tier() -> None:
    cheap = _model("cheap", 2, True, {"ST-12": _strategy("shop", 0.70, 0.29)})
    costly = _model("costly", 3, True, {"ST-12": _strategy("shop", 0.70, 0.29)})

    assert _pick(models=(costly, cheap)) == ("cheap", "ST-12")


def test_unfinished_training_is_never_used() -> None:
    unfinished = _model(
        "rapidocr-linear",
        2,
        True,
        {"ST-12": _strategy("shop", 0.63, 0.29)},
        finished=False,
    )

    assert _pick(models=(PRIORS, unfinished)) == ("priors", "ST-19")
