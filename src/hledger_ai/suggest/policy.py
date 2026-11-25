"""Which model and strategy answers a field (EV-3, EV-4, SP-5).

Every finished model has an accuracy per strategy and field, measured on
receipts it never saw. For a field the most accurate usable option wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set

from typeguard import typechecked

from hledger_ai.strategies import STRATEGIES
from hledger_ai.training.state import ModelState


@typechecked
@dataclass(frozen=True)
class ModelView:
    name: str
    tier: int
    uses_image: bool
    state: ModelState


@typechecked
@dataclass(frozen=True)
class Choice:
    model: str
    strategy_id: str
    accuracy: float
    tier: int


@typechecked
def options(
    *,
    field: str,
    known: Set[str],
    has_image: bool,
    models: Sequence[ModelView],
    budget_seconds: Callable[[str], float],
) -> List[Choice]:
    """Usable (model, strategy) pairs for *field*, best first.

    Usable means: trained to the end, every field the strategy needs is
    confirmed, a photo is there when the model reads photos, the model
    beats the baseline (EV-3; a tier-1 prior may equal it, being the
    baseline itself) and answers within the time budget (SP-5).
    """
    found: List[Choice] = []
    for model in models:
        if not model.state.finished:
            continue
        if model.uses_image and not has_image:
            continue
        for strategy in STRATEGIES:
            progress = model.state.strategies.get(strategy.id)
            if progress is None or field not in strategy.improves:
                continue
            needed = {k for k in strategy.known if k != "image"}
            if not needed <= known:
                continue
            accuracy = progress.field_accuracy(field=field)
            if accuracy is None:
                continue
            baseline = _field_baseline(progress=progress, field=field)
            beats = accuracy > baseline or (
                model.tier == 1 and accuracy >= baseline
            )
            seconds = progress.seconds_per_receipt() or 0.0
            if beats and seconds <= budget_seconds(strategy.id):
                found.append(
                    Choice(
                        model=model.name,
                        strategy_id=strategy.id,
                        accuracy=accuracy,
                        tier=model.tier,
                    )
                )
    return sorted(found, key=lambda c: (-c.accuracy, c.tier, c.model))


@typechecked
def _field_baseline(*, progress, field: str) -> float:
    total = sum(
        int(fields.get(field, {}).get("total", 0))
        for fields in progress.folds.values()
    )
    hits = sum(
        int(fields.get(field, {}).get("baseline_correct", 0))
        for fields in progress.folds.values()
    )
    return hits / total if total else 0.0


@typechecked
def choose(
    *,
    field: str,
    known: Set[str],
    has_image: bool,
    models: Sequence[ModelView],
    budget_seconds: Callable[[str], float],
) -> Optional[Choice]:
    """EV-4: the best usable option, or None."""
    ranked = options(
        field=field,
        known=known,
        has_image=has_image,
        models=models,
        budget_seconds=budget_seconds,
    )
    return ranked[0] if ranked else None
