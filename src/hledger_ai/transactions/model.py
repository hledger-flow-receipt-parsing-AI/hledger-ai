"""``csv-linear``: TF-IDF over the text of a bank row (character and word
n-grams) plus a few tokens (direction, own account, amount size) into a
logistic regression.  Light enough for any laptop; trains in seconds."""

from __future__ import annotations

import math

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
from hledger_ai.transactions.dataset import TransactionExample
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typeguard import typechecked

from hledger_core.categorisation.rules import RowView

ENGINE_VERSION = "1"  # bump when the features change: a plain --train retrains
# lbfgs converges long before this on TF-IDF features; the cap keeps a
# slow-converging fit (hundreds of categories) to minutes, not tens.
MAX_ITER = 800
Ranked = List[Tuple[str, float]]


@typechecked
def example_text(row: RowView) -> str:
    return " | ".join(
        part
        for part in (row.payee, row.description, row.counter_account)
        if part
    )


@typechecked
def amount_bucket(amount: float) -> int:
    """Amounts of the same order of magnitude share a bucket."""
    return int(math.log2(abs(amount) + 1.0))


@typechecked
def example_tokens(row: RowView) -> Dict[str, float]:
    return {
        f"direction={row.direction}": 1.0,
        f"account={row.own_account}": 1.0,
        f"amount_bucket={amount_bucket(row.amount)}": 1.0,
    }


class NotEnoughData(ValueError):
    """Fewer than two categories to learn from."""


@typechecked
class TransactionModel:
    def __init__(self, *, regularisation: float = 10.0) -> None:
        self._regularisation = regularisation
        self._chars: Optional[TfidfVectorizer] = None
        self._words: Optional[TfidfVectorizer] = None
        self._tokens: Optional[DictVectorizer] = None
        self._model: Optional[LogisticRegression] = None
        self._only_class: Optional[str] = None
        self.version = ENGINE_VERSION

    # ------------------------------------------------------------- fit
    def fit(
        self, *, examples: Sequence[TransactionExample]
    ) -> "TransactionModel":
        labels = [e.label for e in examples]
        classes = sorted(set(labels))
        if not classes:
            raise NotEnoughData("No categorised transactions to learn from")
        if len(classes) == 1:
            self._only_class = classes[0]
            return self
        texts = [example_text(e.row) for e in examples]
        self._chars = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True,
            max_features=80000,
        ).fit(texts)
        self._words = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            token_pattern=r"[^\s|]+",
        ).fit(texts)
        self._tokens = DictVectorizer(sparse=True).fit(
            [example_tokens(e.row) for e in examples]
        )
        self._model = LogisticRegression(
            C=self._regularisation, max_iter=MAX_ITER
        ).fit(self._matrix(rows=[e.row for e in examples]), labels)
        # Hundreds of categories times tens of thousands of features: half
        # the file and the memory, no measurable change in the ranking.
        self._model.coef_ = self._model.coef_.astype(np.float32)
        self._model.intercept_ = self._model.intercept_.astype(np.float32)
        return self

    def _matrix(self, *, rows: Sequence[RowView]) -> Any:
        texts = [example_text(r) for r in rows]
        return hstack(
            [
                self._chars.transform(texts),
                self._words.transform(texts),
                self._tokens.transform([example_tokens(r) for r in rows]),
            ]
        ).tocsr()

    @property
    def is_trained(self) -> bool:
        return self._model is not None or self._only_class is not None

    @property
    def classes(self) -> List[str]:
        if self._model is not None:
            return [str(c) for c in self._model.classes_]
        return [self._only_class] if self._only_class else []

    # --------------------------------------------------------- suggest
    def probabilities(self, *, rows: Sequence[RowView]) -> List[float]:
        """Mean class probabilities over *rows*, in ``classes`` order."""
        if self._model is None:
            return [1.0] if self._only_class else []
        matrix = self._matrix(rows=rows)
        return [
            float(p) for p in self._model.predict_proba(matrix).mean(axis=0)
        ]

    def suggest_group(
        self, *, rows: Sequence[RowView], limit: int = 5
    ) -> Ranked:
        if not rows or not self.is_trained:
            return []
        ranked = sorted(
            zip(self.classes, self.probabilities(rows=rows)),
            key=lambda cp: (-cp[1], cp[0]),
        )
        return [(label, probability) for label, probability in ranked[:limit]]

    def suggest(self, *, row: RowView, limit: int = 5) -> Ranked:
        return self.suggest_group(rows=[row], limit=limit)

    # ------------------------------------------------------------ disk
    def save(self, *, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(*, path: str) -> "TransactionModel":
        model = joblib.load(path)
        if not isinstance(model, TransactionModel):
            raise TypeError(f"{path} does not hold a TransactionModel")
        return model


@typechecked
class MostFrequent:
    """EV-3: the baseline every fold is compared with."""

    def __init__(self) -> None:
        self.label: Optional[str] = None

    def fit(self, *, examples: Sequence[TransactionExample]) -> "MostFrequent":
        counts: Dict[str, int] = {}
        for example in examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        self.label = (
            max(sorted(counts), key=lambda k: counts[k]) if counts else None
        )
        return self


@typechecked
def score(
    *,
    model: TransactionModel,
    baseline: MostFrequent,
    test: Sequence[TransactionExample],
) -> Dict[str, int]:
    """Top-1 hits of the model and of the baseline on *test*."""
    correct = 0
    for example in test:
        ranked = model.suggest(row=example.row, limit=1)
        if ranked and ranked[0][0] == example.label:
            correct += 1
    return {
        "correct": correct,
        "total": len(test),
        "baseline_correct": sum(1 for e in test if e.label == baseline.label),
    }
