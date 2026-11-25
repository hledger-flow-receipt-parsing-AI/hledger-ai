"""Accuracy per strategy and field on unseen receipts (EV-1, EV-3)."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Hashable, List, Optional, Sequence, Set, Tuple

from typeguard import typechecked

from hledger_ai.models.experts import TrainingExample
from hledger_ai.receipt_fields import (
    FIELDS,
    ReceiptFacts,
    known_from_facts,
    known_key,
)
from hledger_ai.strategies import STRATEGIES, Strategy
from hledger_ai.suggest.engine import SuggestionEngine, image_strategy_id

_MONEY_FIELDS = ("amount", "change", "subtotal", "total_tax")


@typechecked
@dataclass
class FieldScore:
    correct: int = 0
    total: int = 0
    baseline_correct: int = 0
    seconds: float = 0.0  # time the model spent on these receipts (SP-5)


@typechecked
def comparable(*, field: str, value: Any) -> Optional[Hashable]:
    """A value in the form two answers to the same question compare in."""
    if value is None or value == "":
        return None
    if field in _MONEY_FIELDS:
        return int(round(float(value) * 100))
    if isinstance(value, bool):
        return "y" if value else "n"
    if isinstance(value, datetime):
        return value.date() if field == "date" else value
    if isinstance(value, date):
        return value
    return str(value)


@typechecked
def _known_for(
    *, strategy: Strategy, facts: ReceiptFacts
) -> Optional[Dict[str, Any]]:
    """The strategy's known fields from the label; None if one is missing."""
    labelled = known_from_facts(facts=facts)
    required = [field for field in strategy.known if field != "image"]
    if any(field not in labelled for field in required):
        return None
    return {field: labelled[field] for field in required}


@typechecked
class Baseline:
    """EV-3: the most frequent value given the strategy's known fields."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[str, str], Dict[Tuple, Counter]] = {}

    @staticmethod
    def _key(*, known: Dict[str, Any]) -> Tuple:
        return tuple(
            known_key(known_field=field, value=value)
            for field, value in sorted(known.items())
        )

    @staticmethod
    def fit(*, examples: Sequence[TrainingExample]) -> "Baseline":
        baseline = Baseline()
        for strategy in STRATEGIES:
            for field in strategy.improves:
                if field not in FIELDS:
                    continue
                table: Dict[Tuple, Counter] = defaultdict(Counter)
                for example in examples:
                    known = _known_for(strategy=strategy, facts=example.facts)
                    target = comparable(
                        field=field, value=example.facts.get(field=field)
                    )
                    if known is None or target is None:
                        continue
                    table[Baseline._key(known=known)][target] += 1
                    table[("<any>",)][target] += 1
                baseline._counts[(strategy.id, field)] = table
        return baseline

    def predict(
        self, *, strategy: Strategy, field: str, facts: ReceiptFacts
    ) -> Optional[Hashable]:
        table = self._counts.get((strategy.id, field), {})
        known = _known_for(strategy=strategy, facts=facts)
        seen = table.get(Baseline._key(known=known or {})) or table.get(
            ("<any>",)
        )
        if not seen:
            return None
        return min(seen, key=lambda value: (-seen[value], str(value)))


@typechecked
def scorable_strategies(
    *, use_image: bool, strategy_ids: Optional[Set[str]] = None
) -> List[Strategy]:
    """Strategies a model of this kind can be scored on."""
    return [
        strategy
        for strategy in STRATEGIES
        if strategy.applicable
        and (strategy_ids is None or strategy.id in strategy_ids)
        and (use_image or "image" not in strategy.known)
        # With a photo, "nothing known" is just the image-only strategy.
        and not (use_image and not strategy.known)
        and any(field in FIELDS for field in strategy.improves)
    ]


@typechecked
def score_engine(
    *,
    engine: SuggestionEngine,
    baseline: Baseline,
    test: Sequence[TrainingExample],
    use_image: bool,
    strategy_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, FieldScore]]:
    """Score every strategy of *engine* on receipts it never saw.

    Returns ``{strategy id: {field: score}}`` for strategies that could be
    scored on at least one receipt.
    """
    scores: Dict[str, Dict[str, FieldScore]] = {}
    for strategy in scorable_strategies(
        use_image=use_image, strategy_ids=strategy_ids
    ):
        active = {strategy.id}
        if use_image:
            active |= {
                image_strategy_id(field=f) or "" for f in strategy.improves
            }
        fields = [f for f in strategy.improves if f in FIELDS]
        per_field: Dict[str, FieldScore] = defaultdict(FieldScore)
        for example in test:
            known = _known_for(strategy=strategy, facts=example.facts)
            if known is None:
                continue
            started = time.monotonic()
            suggestions = engine.suggest(
                evidence=example.evidence if use_image else None,
                known=known,
                strategies=active,
                fields=set(fields),
            )
            elapsed = time.monotonic() - started
            for field in fields:
                truth = comparable(
                    field=field, value=example.facts.get(field=field)
                )
                if truth is None:
                    continue
                score = per_field[field]
                score.total += 1
                score.seconds += elapsed
                top = suggestions.get(field)
                if top and comparable(field=field, value=top[0].value) == truth:
                    score.correct += 1
                guess = baseline.predict(
                    strategy=strategy, field=field, facts=example.facts
                )
                if guess == truth:
                    score.baseline_correct += 1
        if per_field:
            scores[strategy.id] = dict(per_field)
    return scores


@typechecked
def needed_fields(*, strategies: Sequence[Strategy]) -> Set[str]:
    """Receipt fields the given strategies improve."""
    return {f for s in strategies for f in s.improves if f in FIELDS}


@typechecked
def evaluate_fold(
    *,
    train: Sequence[TrainingExample],
    test: Sequence[TrainingExample],
    use_image: bool,
    strategy_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, FieldScore]]:
    """Train on *train*, score every strategy on *test*."""
    wanted = scorable_strategies(use_image=use_image, strategy_ids=strategy_ids)
    engine = SuggestionEngine.train(
        examples=train,
        use_image=use_image,
        fields=needed_fields(strategies=wanted),
    )
    return score_engine(
        engine=engine,
        baseline=Baseline.fit(examples=train),
        test=test,
        use_image=use_image,
        strategy_ids=strategy_ids,
    )
