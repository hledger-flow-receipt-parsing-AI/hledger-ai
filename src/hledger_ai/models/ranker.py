"""Tier-2 candidate ranker (TR-1): a small linear model that learns which
of the candidates found on a receipt is the labelled value."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from typeguard import typechecked


@typechecked
class CandidateRanker:
    """Pointwise ranker: P(this candidate is the right one)."""

    def __init__(self, *, regularisation: float = 3.0) -> None:
        self._regularisation = regularisation
        self._vectorizer: Optional[DictVectorizer] = None
        self._model: Optional[LogisticRegression] = None
        # Set when every training candidate was right: nothing to learn,
        # any candidate is as good as the examples.
        self._always_right = False

    @property
    def is_trained(self) -> bool:
        return self._model is not None or self._always_right

    def fit(
        self,
        *,
        groups: Sequence[Sequence[Dict[str, float]]],
        correct: Sequence[Sequence[bool]],
        weights: Optional[Sequence[float]] = None,
    ) -> "CandidateRanker":
        """*groups* holds the candidates of each receipt; *correct* marks
        the labelled ones; *weights* optionally weighs whole receipts."""
        rows: List[Dict[str, float]] = []
        targets: List[int] = []
        row_weights: List[float] = []
        for index, (group, marks) in enumerate(zip(groups, correct)):
            for features, mark in zip(group, marks):
                rows.append(features)
                targets.append(int(mark))
                row_weights.append(weights[index] if weights else 1.0)
        if len(set(targets)) < 2:
            self._always_right = bool(targets) and targets[0] == 1
            return self  # nothing to tell apart yet
        self._vectorizer = DictVectorizer(sparse=True)
        matrix = self._vectorizer.fit_transform(rows)
        self._model = LogisticRegression(C=self._regularisation, max_iter=2000)
        self._model.fit(matrix, targets, sample_weight=row_weights)
        return self

    def probabilities(
        self, *, features: Sequence[Dict[str, float]]
    ) -> List[float]:
        if not features:
            return []
        if self._model is None or self._vectorizer is None:
            return [1.0 if self._always_right else 0.0] * len(features)
        matrix = self._vectorizer.transform(list(features))
        return [float(p) for p in self._model.predict_proba(matrix)[:, 1]]


@typechecked
def rank_values(
    *, values: Sequence[Hashable], probabilities: Sequence[float]
) -> List[Tuple[Any, float]]:
    """Merge occurrences of the same value and rank the values.

    A value printed several times is more likely right: its occurrences
    combine as independent evidence. Confidences are scaled down when they
    would add up to more than one.
    """
    miss: Dict[Hashable, float] = {}
    for value, probability in zip(values, probabilities):
        miss[value] = miss.get(value, 1.0) * (1.0 - probability)
    merged = {value: 1.0 - m for value, m in miss.items()}
    total = sum(merged.values())
    scale = total if total > 1.0 else 1.0
    return sorted(
        ((value, p / scale) for value, p in merged.items()),
        key=lambda vp: -vp[1],
    )
