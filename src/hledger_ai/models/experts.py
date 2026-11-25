"""Tier-2 experts (TR-1): one small trained model per receipt field.

Each expert makes a first guess from the receipt image alone (CI-2) and
returns ranked candidates with a confidence (CI-3). Some can also use a
field the user already confirmed (CI-1), passed in ``known``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from typeguard import typechecked

from hledger_ai.candidates.amounts import AmountCandidate
from hledger_ai.models.features import (
    amount_features,
    date_features,
    time_features,
)
from hledger_ai.models.ranker import CandidateRanker, rank_values
from hledger_ai.models.text_features import TextFeaturizer
from hledger_ai.receipt_fields import (
    OPTIONAL_AMOUNTS,
    ReceiptFacts,
    known_from_facts,
    known_key,
)
from hledger_ai.suggest.evidence import ReceiptEvidence

Ranked = List[Tuple[Any, float]]

# Fields that only some receipts have: that one exists at all says something
# (a receipt linked to a bank transaction was not paid in cash).
_PRESENCE_MATTERS = ("csv_transaction",)


@typechecked
@dataclass(frozen=True)
class TrainingExample:
    facts: ReceiptFacts
    evidence: ReceiptEvidence
    weight: float = 1.0  # TR-4: corrected receipts weigh more


@typechecked
def _cents(*, value: float) -> int:
    return int(round(value * 100))


@typechecked
def _reference_features(
    *, cents: int, reference: Optional[int], values: Sequence[int]
) -> Dict[str, float]:
    """How a candidate relates to an amount that is already known."""
    if reference is None or reference <= 0:
        return {}
    ratio = cents / reference
    features = {
        "ref:equal": float(cents == reference),
        "ref:greater": float(cents > reference),
        "ref:complement_printed": float((reference - cents) in values),
    }
    if 0 < ratio <= 0.3:
        features[f"ref:share:{int(ratio * 40)}"] = 1.0  # 2.5% steps
    return features


@typechecked
class AmountExpert:
    """Picks one of the amounts printed on the receipt for *field*.

    With *reference* set, the candidate is also compared with a known
    amount: ``"amount"`` (what was handed over) or ``"total"`` (handed over
    minus change).
    """

    def __init__(self, *, field: str, reference: Optional[str] = None) -> None:
        self.field = field
        self.reference = reference
        self._ranker = CandidateRanker()

    def _reference_cents(self, *, known: Dict[str, Any]) -> Optional[int]:
        amount = known.get("amount")
        if self.reference is None or amount is None:
            return None
        if self.reference == "total":
            return _cents(value=amount - (known.get("change") or 0.0))
        return _cents(value=amount)

    def _features(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        candidates: Sequence[AmountCandidate] = evidence.amounts()
        base = evidence.memo(
            key="amount_features",
            compute=lambda: [
                amount_features(candidate=c, candidates=candidates)
                for c in candidates
            ],
        )
        reference = self._reference_cents(known=known)
        if reference is None:
            return base
        values = [_cents(value=c.value) for c in candidates]
        rows = []
        for candidate, shared in zip(candidates, base):
            features = dict(shared)
            features.update(
                _reference_features(
                    cents=_cents(value=candidate.value),
                    reference=reference,
                    values=values,
                )
            )
            rows.append(features)
        return rows

    def _target(self, *, facts: ReceiptFacts) -> Optional[float]:
        return facts.get(field=self.field)

    def fit(self, *, examples: Sequence[TrainingExample]) -> "AmountExpert":
        groups, correct, weights = [], [], []
        for example in examples:
            target = self._target(facts=example.facts)
            if target is None:
                continue
            candidates = example.evidence.amounts()
            known = {
                "amount": example.facts.amount,
                "change": example.facts.change,
            }
            groups.append(
                self._features(evidence=example.evidence, known=known)
            )
            correct.append(
                [
                    _cents(value=c.value) == _cents(value=target)
                    for c in candidates
                ]
            )
            weights.append(example.weight)
        self._ranker = CandidateRanker().fit(
            groups=groups, correct=correct, weights=weights
        )
        return self

    def suggest(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> Ranked:
        candidates = evidence.amounts()
        if not candidates or not self._ranker.is_trained:
            return []
        ranked = rank_values(
            values=[_cents(value=c.value) for c in candidates],
            probabilities=self._ranker.probabilities(
                features=self._features(evidence=evidence, known=known)
            ),
        )
        return [(cents / 100.0, confidence) for cents, confidence in ranked]


@typechecked
class OptionalAmountExpert:
    """An amount not every receipt has (the subtotal): first whether the
    user enters one at all, then which printed amount it is. "Nothing" is
    ranked as the value None, next to the amounts.

    Only complete labels teach the first part: a partial label lacks the
    field because it was never reached.
    """

    def __init__(self, *, field: str, reference: Optional[str] = None) -> None:
        self.field = field
        self.reference = reference
        self._present = ClassExpert(field=f"has_{field}")
        self._value = AmountExpert(field=field, reference=reference)

    def fit(
        self,
        *,
        examples: Sequence[TrainingExample],
        featurizer: Optional[TextFeaturizer] = None,
    ) -> "OptionalAmountExpert":
        self._present = ClassExpert(field=f"has_{self.field}").fit(
            examples=examples, featurizer=featurizer
        )
        self._value = AmountExpert(
            field=self.field, reference=self.reference
        ).fit(examples=examples)
        return self

    def suggest(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> Ranked:
        values = self._value.suggest(evidence=evidence, known=known)
        chances = dict(self._present.suggest(evidence=evidence, known={}))
        if not chances:
            return values  # no complete label: nothing learnt about absence
        present = chances.get("yes", 0.0)
        ranked: Ranked = [(None, 1.0 - present)]
        ranked += [
            (value, confidence * present) for value, confidence in values
        ]
        return sorted(ranked, key=lambda vc: -vc[1])


@typechecked
class ChangeExpert:
    """Change returned: zero for most card payments, otherwise one of the
    printed amounts. Uses the payment account, the category and the amount
    handed over when they are known."""

    field = "change"

    def __init__(self, *, context: Tuple[str, ...] = ()) -> None:
        self.context = context
        self._any_change = ClassExpert(field="has_change", context=context)
        self._value = AmountExpert(field="change")
        self._value_given_amount = AmountExpert(
            field="change", reference="amount"
        )

    def fit(
        self,
        *,
        examples: Sequence[TrainingExample],
        featurizer: Optional[TextFeaturizer] = None,
    ) -> "ChangeExpert":
        labelled = [e for e in examples if e.facts.change is not None]
        self._any_change = ClassExpert(
            field="has_change", context=self.context
        ).fit(examples=labelled, featurizer=featurizer)
        with_change = [e for e in labelled if e.facts.change > 0]
        self._value = AmountExpert(field="change").fit(examples=with_change)
        self._value_given_amount = AmountExpert(
            field="change", reference="amount"
        ).fit(examples=with_change)
        return self

    def suggest(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> Ranked:
        chances = dict(self._any_change.suggest(evidence=evidence, known=known))
        if not chances:
            return []
        some = chances.get("yes", 0.0)
        expert = (
            self._value_given_amount
            if known.get("amount") is not None
            else self._value
        )
        ranked: Ranked = [(0.0, 1.0 - some)]
        ranked += [
            (value, confidence * some)
            for value, confidence in expert.suggest(
                evidence=evidence, known=known
            )
            if value > 0
        ]
        return sorted(ranked, key=lambda vc: -vc[1])

    def text_scores(
        self, *, texts: Sequence[str], has_change: bool
    ) -> List[float]:
        """How much each text speaks for change being returned, or not."""
        return self._any_change.text_scores(
            texts=texts, value="yes" if has_change else "no"
        )


@typechecked
def _date_rows(*, evidence: ReceiptEvidence) -> List[Dict[str, float]]:
    candidates = evidence.dates()
    return evidence.memo(
        key="date_features",
        compute=lambda: [
            date_features(candidate=c, candidates=candidates)
            for c in candidates
        ],
    )


@typechecked
def _time_rows(*, evidence: ReceiptEvidence) -> List[Dict[str, float]]:
    candidates = evidence.times()
    return evidence.memo(
        key="time_features",
        compute=lambda: [
            time_features(candidate=c, candidates=candidates)
            for c in candidates
        ],
    )


@typechecked
class DateExpert:
    field = "date"

    def __init__(self) -> None:
        self._ranker = CandidateRanker()

    def fit(self, *, examples: Sequence[TrainingExample]) -> "DateExpert":
        groups, correct, weights = [], [], []
        for example in examples:
            target = example.facts.get(field="date")
            if target is None:
                continue
            candidates = example.evidence.dates()
            groups.append(_date_rows(evidence=example.evidence))
            correct.append([c.value == target for c in candidates])
            weights.append(example.weight)
        self._ranker = CandidateRanker().fit(
            groups=groups, correct=correct, weights=weights
        )
        return self

    def suggest(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> Ranked:
        candidates = evidence.dates()
        if not candidates or not self._ranker.is_trained:
            return []
        return rank_values(
            values=[c.value for c in candidates],
            probabilities=self._ranker.probabilities(
                features=_date_rows(evidence=evidence)
            ),
        )


@typechecked
class TimeExpert:
    field = "time"

    def __init__(self) -> None:
        self._ranker = CandidateRanker()

    def fit(self, *, examples: Sequence[TrainingExample]) -> "TimeExpert":
        groups, correct, weights = [], [], []
        for example in examples:
            target = example.facts.get(field="time")
            if target is None:
                continue
            candidates = example.evidence.times()
            groups.append(_time_rows(evidence=example.evidence))
            correct.append([c.value == target for c in candidates])
            weights.append(example.weight)
        self._ranker = CandidateRanker().fit(
            groups=groups, correct=correct, weights=weights
        )
        return self

    def suggest(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> Ranked:
        candidates = evidence.times()
        if not candidates or not self._ranker.is_trained:
            return []
        return rank_values(
            values=[c.value for c in candidates],
            probabilities=self._ranker.probabilities(
                features=_time_rows(evidence=evidence)
            ),
        )


@typechecked
def class_label(*, value: Any) -> Optional[str]:
    """Field value as a class name; yes/no questions become ``y``/``n``."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "y" if value else "n"
    return str(value)


@typechecked
def class_target(*, field: str, facts: ReceiptFacts) -> Optional[str]:
    """The class a label holds for *field*; ``has_change`` is change yes/no
    and ``has_subtotal`` whether a subtotal was entered at all (unknown
    for a partial label)."""
    if field == "has_change":
        if facts.change is None:
            return None
        return "yes" if facts.change > 0 else "no"
    if field.startswith("has_") and field[len("has_") :] in OPTIONAL_AMOUNTS:
        if not facts.complete:
            return None
        return "no" if facts.get(field=field[len("has_") :]) is None else "yes"
    return class_label(value=facts.get(field=field))


@typechecked
def context_tokens(
    *, context: Sequence[str], known: Dict[str, Any]
) -> Dict[str, float]:
    """Confirmed fields as model inputs. A category also counts as each of
    its parents, so a new subcategory still carries information."""
    tokens: Dict[str, float] = {}
    for field in context:
        key = known_key(known_field=field, value=known.get(field))
        if key is None:
            continue
        if field in _PRESENCE_MATTERS:
            tokens[f"{field}?"] = 1.0
        if field == "category":
            parts = key.split(":")
            for depth in range(1, len(parts) + 1):
                tokens[f"category^={':'.join(parts[:depth])}"] = 1.0
        else:
            tokens[f"{field}={key}"] = 1.0
    return tokens


@typechecked
class ClassExpert:
    """Chooses among the values seen in the labels (category, shop, payment
    account, currency, withdrawal yes/no) from the receipt text and from the
    confirmed fields named in *context* (CI-1)."""

    def __init__(
        self,
        *,
        field: str,
        context: Tuple[str, ...] = (),
        regularisation: float = 20.0,
    ) -> None:
        self.field = field
        self.context = context
        self._regularisation = regularisation
        self._featurizer: Optional[TextFeaturizer] = None
        self._context_vectorizer: Optional[DictVectorizer] = None
        self._model: Optional[LogisticRegression] = None
        self._only_class: Optional[str] = None

    def _matrix(self, *, text_rows: List[Any], tokens: List[Dict]) -> Any:
        text = vstack(text_rows)
        if self._context_vectorizer is None:
            return text.tocsr()
        return hstack(
            [text, self._context_vectorizer.transform(tokens)]
        ).tocsr()

    def fit(
        self,
        *,
        examples: Sequence[TrainingExample],
        featurizer: Optional[TextFeaturizer] = None,
    ) -> "ClassExpert":
        rows = [
            (e, class_target(field=self.field, facts=e.facts)) for e in examples
        ]
        rows = [(e, label) for e, label in rows if label is not None]
        self._featurizer = featurizer or TextFeaturizer().fit(
            texts=[e.evidence.text() for e, _ in rows]
        )
        classes = sorted({label for _, label in rows})
        self._only_class = classes[0] if len(classes) == 1 else None
        if len(classes) < 2 or not self._featurizer.is_fitted:
            return self

        # Every receipt is seen with its confirmed fields and without, so
        # the model works before and after the user answers them.
        text_rows, tokens, labels, weights = [], [], [], []
        for example, label in rows:
            full = context_tokens(
                context=self.context,
                known=known_from_facts(facts=example.facts),
            )
            for variant in ([full, {}] if full else [{}]):
                text_rows.append(
                    self._featurizer.vector(evidence=example.evidence)
                )
                tokens.append(variant)
                labels.append(label)
                weights.append(example.weight)
        if any(tokens):
            self._context_vectorizer = DictVectorizer(sparse=True).fit(tokens)
        self._model = LogisticRegression(
            C=self._regularisation, max_iter=2000
        ).fit(
            self._matrix(text_rows=text_rows, tokens=tokens),
            labels,
            sample_weight=weights,
        )
        return self

    def suggest(
        self, *, evidence: ReceiptEvidence, known: Dict[str, Any]
    ) -> Ranked:
        if self._model is None or self._featurizer is None:
            return [(self._only_class, 1.0)] if self._only_class else []
        matrix = self._matrix(
            text_rows=[self._featurizer.vector(evidence=evidence)],
            tokens=[context_tokens(context=self.context, known=known)],
        )
        probabilities = self._model.predict_proba(matrix)[0]
        return sorted(
            (
                (str(c), float(p))
                for c, p in zip(self._model.classes_, probabilities)
            ),
            key=lambda cp: (-cp[1], cp[0]),
        )

    def text_scores(self, *, texts: Sequence[str], value: str) -> List[float]:
        """How much each text (a printed row or box) speaks for *value*:
        its part in the model's score for that class. Zero when untrained
        or when *value* is no class the model knows."""
        nothing = [0.0] * len(texts)
        if (
            not texts
            or self._model is None
            or self._featurizer is None
            or not self._featurizer.is_fitted
        ):
            return nothing
        classes = [str(c) for c in self._model.classes_]
        if value not in classes:
            return nothing
        matrix = self._featurizer.transform(texts=texts)
        # The text columns come first; the context tokens follow them.
        coefficients = self._model.coef_[:, : matrix.shape[1]]
        if coefficients.shape[0] == 1:
            # Two classes: the coefficients are those of the second one.
            sign = 1.0 if classes.index(value) == 1 else -1.0
            weights = coefficients[0] * sign
        else:
            weights = coefficients[classes.index(value)]
        return [float(score) for score in matrix @ weights]
