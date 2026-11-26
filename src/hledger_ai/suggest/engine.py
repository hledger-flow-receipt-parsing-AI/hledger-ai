"""The suggestion engine (requirements CI-1 to CI-4).

For every field the user has not answered yet it returns ranked candidates
with a confidence. Confirmed fields are never suggested again, so a
suggestion can never overwrite them.

Two kinds of engine exist:

* the tier-2 engine holds one trained expert per field; the expert reads the
  receipt image (through OCR) and takes the confirmed fields as extra inputs;
* the tier-1 engine (``use_image=False``) holds frequency priors only and
  needs no image at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import joblib
from typeguard import typechecked

from hledger_ai.models.experts import (
    AmountExpert,
    ChangeExpert,
    ClassExpert,
    DateExpert,
    OptionalAmountExpert,
    Ranked,
    TimeExpert,
    TrainingExample,
    class_target,
)
from hledger_ai.models.priors import FrequencyPrior
from hledger_ai.models.text_features import TextFeaturizer
from hledger_ai.receipt_fields import (
    FIELDS,
    OPTIONAL_AMOUNTS,
    known_from_facts,
    known_key,
)
from hledger_ai.strategies import STRATEGIES, get_strategy
from hledger_ai.suggest.evidence import ReceiptEvidence

# Bump when experts, features or fusion change (DS-2: model version).
# 4: text vectors no longer leak between training folds (TextFeaturizer).
# 5: "nothing" is a subtotal answer (OptionalAmountExpert); manual address
#    knowledge.
# 6: dates after leader dots ("Date... .17/03/25"), "aou", French times.
ENGINE_VERSION = "6"

# The strategy that knows nothing: the most frequent value so far.
USUAL_VALUE = "ST-19"

# Confidence given to a value no prior proposed yet, so a later prior can
# still bring it up.
_UNSEEN_FLOOR = 0.02


@typechecked
@dataclass(frozen=True)
class Suggestion:
    value: Any
    confidence: float
    sources: Tuple[str, ...]  # strategy ids that produced it


@typechecked
@dataclass(frozen=True)
class CrossSpec:
    """One cross-inference link behind a strategy: a confirmed field that
    improves another field."""

    strategy_id: str
    known_field: str
    improves: str  # a receipt field, or "has_change" (change yes/no)
    hierarchical: bool = False


CROSS_SPECS: Tuple[CrossSpec, ...] = (
    CrossSpec("ST-1", "category", "shop", hierarchical=True),
    CrossSpec("ST-2", "shop", "category"),
    CrossSpec("ST-3", "shop_name", "shop"),
    CrossSpec("ST-4", "address", "shop"),
    CrossSpec("ST-6", "csv_transaction", "shop"),
    CrossSpec("ST-6", "csv_transaction", "category"),
    CrossSpec("ST-6", "csv_transaction", "bank_account"),
    CrossSpec("ST-16", "category", "bank_account", hierarchical=True),
    CrossSpec("ST-17", "bank_account", "currency"),
    CrossSpec("ST-17", "bank_account", "has_change"),
)


@typechecked
def _target_name(*, field: str) -> str:
    return "has_change" if field == "change" else field


@typechecked
def _specs_improving(*, field: str) -> List[CrossSpec]:
    target = _target_name(field=field)
    return [spec for spec in CROSS_SPECS if spec.improves == target]


@typechecked
def _context(*, field: str) -> Tuple[str, ...]:
    """Confirmed fields the expert of *field* takes as inputs."""
    return tuple(spec.known_field for spec in _specs_improving(field=field))


@typechecked
def image_strategy_id(*, field: str) -> Optional[str]:
    """The strategy that guesses *field* from the image alone (CI-2)."""
    for strategy in STRATEGIES:
        if strategy.known == ("image",) and field in strategy.improves:
            return strategy.id
    return None


@typechecked
def _amount_expert(*, field: str, reference: Optional[str] = None) -> Any:
    """The expert of an amount field; one a receipt may lack (the
    subtotal) can also answer "nothing"."""
    if field in OPTIONAL_AMOUNTS:
        return OptionalAmountExpert(field=field, reference=reference)
    return AmountExpert(field=field, reference=reference)


@typechecked
def _build_experts() -> Dict[str, Any]:
    experts: Dict[str, Any] = {
        "date": DateExpert(),
        "time": TimeExpert(),
        "amount": AmountExpert(field="amount"),
        "change": ChangeExpert(context=_context(field="change")),
        "subtotal": _amount_expert(field="subtotal"),
        "total_tax": _amount_expert(field="total_tax"),
        # ST-20: the same fields once the paid total is known.
        "subtotal|total": _amount_expert(field="subtotal", reference="total"),
        "total_tax|total": _amount_expert(field="total_tax", reference="total"),
    }
    for field in (
        "is_withdrawal",
        "category",
        "bank_account",
        "currency",
        "shop",
    ):
        experts[field] = ClassExpert(field=field, context=_context(field=field))
    return experts


@typechecked
class SuggestionEngine:
    def __init__(
        self,
        *,
        experts: Dict[str, Any],
        priors: Dict[Tuple[str, str], FrequencyPrior],
        address: Any = None,
    ) -> None:
        self._experts = experts  # field (or "field|reference") -> expert
        self._priors = priors  # (strategy id, improved field) -> prior
        # address_parts.AddressKnowledge: the manual address questions
        # (CI-6); None in engines trained before it existed.
        self._address = address

    @property
    def uses_image(self) -> bool:
        return bool(self._experts)

    @property
    def address(self) -> Optional[Any]:
        """What the labels say about shop addresses (AddressKnowledge)."""
        return self._address

    def expert(self, *, field: str) -> Optional[Any]:
        """The tier-2 expert of *field*, if this engine has one."""
        return self._experts.get(field)

    @staticmethod
    def train(
        *,
        examples: Sequence[TrainingExample],
        use_image: bool = True,
        fields: Optional[Set[str]] = None,
    ) -> "SuggestionEngine":
        """Fit every expert (tier 2), or only the priors (tier 1) with
        ``use_image=False``. *fields* limits the work to those fields."""

        def wanted(field: str) -> bool:
            return fields is None or field in fields

        experts: Dict[str, Any] = {}
        priors: Dict[Tuple[str, str], FrequencyPrior] = {}
        if use_image and examples:
            featurizer = TextFeaturizer().fit(
                texts=[e.evidence.text() for e in examples]
            )
            for name, expert in _build_experts().items():
                if not wanted(name.split("|")[0]):
                    continue
                if isinstance(
                    expert, (ClassExpert, ChangeExpert, OptionalAmountExpert)
                ):
                    expert.fit(examples=examples, featurizer=featurizer)
                else:
                    expert.fit(examples=examples)
                experts[name] = expert
        if not use_image:
            for spec in CROSS_SPECS:
                if not wanted(
                    "change" if spec.improves == "has_change" else spec.improves
                ):
                    continue
                pairs = []
                for example in examples:
                    known = known_key(
                        known_field=spec.known_field,
                        value=known_from_facts(facts=example.facts).get(
                            spec.known_field
                        ),
                    )
                    target = class_target(
                        field=spec.improves, facts=example.facts
                    )
                    if known is not None and target is not None:
                        pairs.append((known, target))
                if pairs:
                    priors[(spec.strategy_id, spec.improves)] = (
                        FrequencyPrior.fit(
                            pairs=pairs, hierarchical=spec.hierarchical
                        )
                    )
            for field in get_strategy(strategy_id=USUAL_VALUE).improves:
                if not wanted(field):
                    continue
                target = _target_name(field=field)
                usual = [
                    ("", value)
                    for value in (
                        class_target(field=target, facts=e.facts)
                        for e in examples
                    )
                    if value is not None
                ]
                if usual:
                    priors[(USUAL_VALUE, target)] = FrequencyPrior.fit(
                        pairs=usual
                    )
        address = None
        if wanted("shop") and examples:
            # Imported here: address_parts uses this module's Suggestion.
            from hledger_ai.address_parts import AddressKnowledge

            address = AddressKnowledge.fit(examples=examples)
        return SuggestionEngine(experts=experts, priors=priors, address=address)

    def _from_image(
        self,
        *,
        field: str,
        evidence: ReceiptEvidence,
        known: Dict[str, Any],
        enabled: Callable[[str], bool],
    ) -> Tuple[Ranked, List[str]]:
        """Guess of the tier-2 expert, and the strategies behind it."""
        knows_amount = known.get("amount") is not None
        informed = self._experts.get(f"{field}|total")
        if informed is not None and knows_amount and enabled("ST-20"):
            return informed.suggest(evidence=evidence, known=known), ["ST-20"]
        expert = self._experts.get(field)
        strategy_id = image_strategy_id(field=field)
        if expert is None or strategy_id is None:
            return [], []

        used: Dict[str, Any] = {}
        sources: List[str] = [strategy_id] if enabled(strategy_id) else []
        for spec in _specs_improving(field=field):
            if spec.known_field in known and enabled(spec.strategy_id):
                used[spec.known_field] = known[spec.known_field]
                if spec.strategy_id not in sources:
                    sources.append(spec.strategy_id)
        if field == "change" and knows_amount and enabled("ST-18"):
            used["amount"] = known["amount"]
            sources.append("ST-18")
        if not sources:
            return [], []
        return expert.suggest(evidence=evidence, known=used), sources

    def _from_priors(
        self,
        *,
        field: str,
        known: Dict[str, Any],
        enabled: Callable[[str], bool],
    ) -> Tuple[Ranked, List[str]]:
        """Tier 1: what the confirmed fields alone say about *field*."""
        usable = []
        for spec in _specs_improving(field=field):
            prior = self._priors.get((spec.strategy_id, spec.improves))
            key = known_key(
                known_field=spec.known_field,
                value=known.get(spec.known_field),
            )
            if (
                prior is not None
                and key is not None
                and enabled(spec.strategy_id)
            ):
                usable.append((spec, prior, key))
        sources: List[str] = []
        if usable:
            ranked = _combine_priors(priors=[(p, k) for _, p, k in usable])
            for spec, _, _ in usable:
                if spec.strategy_id not in sources:
                    sources.append(spec.strategy_id)
        else:
            usual = self._priors.get((USUAL_VALUE, _target_name(field=field)))
            if usual is None or not enabled(USUAL_VALUE):
                return [], []
            ranked = usual.ranked(known="")
            sources.append(USUAL_VALUE)
        if field == "change":
            # A prior knows whether change is likely, not how much.
            ranked = [(0.0, dict(ranked).get("no", 0.0))]
        return ranked, sources

    def suggest(
        self,
        *,
        evidence: Optional[ReceiptEvidence],
        known: Dict[str, Any],
        strategies: Optional[Set[str]] = None,
        fields: Optional[Set[str]] = None,
        limit: int = 5,
    ) -> Dict[str, List[Suggestion]]:
        """Ranked suggestions for every field not in *known*.

        *strategies* limits the engine to those strategy ids (to score one
        strategy on its own); *fields* limits the work to those fields.
        """

        def enabled(strategy_id: str) -> bool:
            return strategies is None or strategy_id in strategies

        result: Dict[str, List[Suggestion]] = {}
        for field in FIELDS:
            if field in known:
                continue  # CI-4
            if fields is not None and field not in fields:
                continue
            if self.uses_image and evidence is not None:
                ranked, sources = self._from_image(
                    field=field, evidence=evidence, known=known, enabled=enabled
                )
            else:
                ranked, sources = self._from_priors(
                    field=field, known=known, enabled=enabled
                )
            if ranked:
                result[field] = [
                    Suggestion(
                        value=value,
                        confidence=min(max(confidence, 0.0), 1.0),
                        sources=tuple(sources),
                    )
                    for value, confidence in ranked[:limit]
                ]
        return result

    def save(self, *, path: str) -> None:
        joblib.dump(
            {
                "version": ENGINE_VERSION,
                "experts": self._experts,
                "priors": {k: p.to_json() for k, p in self._priors.items()},
                "address": (
                    None if self._address is None else self._address.to_json()
                ),
            },
            path,
        )

    @staticmethod
    def load(*, path: str) -> "SuggestionEngine":
        data = joblib.load(path)
        address = None
        if data.get("address") is not None:
            from hledger_ai.address_parts import AddressKnowledge

            address = AddressKnowledge.from_json(data=data["address"])
        return SuggestionEngine(
            experts=data["experts"],
            priors={
                k: FrequencyPrior.from_json(data=p)
                for k, p in data["priors"].items()
            },
            address=address,
        )


@typechecked
def _combine_priors(*, priors: List[Tuple[FrequencyPrior, str]]) -> Ranked:
    """Overall frequencies times what each confirmed field adds."""
    scores: Dict[Any, float] = dict(priors[0][0].marginal())
    for prior, key in priors:
        marginal = prior.marginal()
        conditional = prior.distribution(known=key)
        for target in conditional:
            if target not in scores:
                scores[target] = _UNSEEN_FLOOR * marginal[target]
        for target in list(scores):
            if target in conditional:
                scores[target] *= conditional[target] / marginal[target]
    total = sum(scores.values())
    if total <= 0:
        return []
    return sorted(
        ((value, score / total) for value, score in scores.items()),
        key=lambda vs: (-vs[1], str(vs[0])),
    )
