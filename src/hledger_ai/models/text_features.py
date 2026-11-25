"""Receipt text as features: character n-grams, so a misread letter still
leaves most of a word recognisable."""

from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from typeguard import typechecked

from hledger_ai.suggest.evidence import ReceiptEvidence


@typechecked
class TextFeaturizer:
    """Fitted once per training set and shared by every class expert, so a
    receipt's text is turned into a vector only once."""

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._token = uuid.uuid4().hex

    def fit(self, *, texts: Sequence[str]) -> "TextFeaturizer":
        # Vectors remembered under another token are another vocabulary's.
        # (Not id(self): Python gives freed ids to new objects, and every
        # training fold makes a new featurizer.)
        self._token = uuid.uuid4().hex
        if any(text.strip() for text in texts):
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                lowercase=True,
                sublinear_tf=True,
                max_features=60000,
            ).fit(list(texts))
        return self

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None

    def vector(self, *, evidence: ReceiptEvidence) -> Any:
        """Sparse row vector of the receipt text."""
        # Models saved before the token existed get one when loaded.
        token = self.__dict__.setdefault("_token", uuid.uuid4().hex)
        return evidence.memo(
            key=("text_vector", token),
            compute=lambda: self._vectorizer.transform([evidence.text()]),
        )
