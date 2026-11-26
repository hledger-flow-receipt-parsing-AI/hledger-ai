"""Receipt text vectors belong to the featurizer that made them.

Training builds a new featurizer per fold. Python may give a new object the
id of one that was freed, so a vector remembered under ``id(featurizer)``
could be handed to a later featurizer with another vocabulary.
"""

import gc

import pytest
from helpers import make_evidence

from hledger_ai.models.text_features import TextFeaturizer


def _featurizer_reusing_id(*, freed_id: int) -> TextFeaturizer:
    """A featurizer that got *freed_id* (CPython hands out freed memory
    again, usually right away)."""
    keep = []
    for _ in range(1000):
        candidate = TextFeaturizer()
        if id(candidate) == freed_id:
            return candidate
        keep.append(candidate)  # hold it, so the next one gets other memory
    # The allocator decides; when other objects of the test run took that
    # memory the situation this test is about cannot be produced here.
    pytest.skip("CPython did not hand the freed id out again in this run")


def test_a_new_featurizer_does_not_get_vectors_of_a_freed_one() -> None:
    evidence = make_evidence(rows=[("Bakkerij Voorbeeld",), ("Totaal 2,50",)])
    first = TextFeaturizer().fit(texts=["ab"])
    first.vector(evidence=evidence)
    freed_id = id(first)
    del first
    gc.collect()

    second = _featurizer_reusing_id(freed_id=freed_id)
    second.fit(texts=["Bakkerij Voorbeeld Totaal 2,50 en nog wat"])

    vocabulary = len(second._vectorizer.vocabulary_)
    assert second.vector(evidence=evidence).shape[1] == vocabulary


def test_a_featurizer_saved_before_tokens_existed_still_works() -> None:
    import pickle

    featurizer = TextFeaturizer().fit(texts=["Bakkerij Voorbeeld Totaal"])
    del featurizer.__dict__["_token"]  # as pickled by engine version 3
    loaded = pickle.loads(pickle.dumps(featurizer))
    evidence = make_evidence(rows=[("Bakkerij Voorbeeld",)])

    vocabulary = len(loaded._vectorizer.vocabulary_)
    assert loaded.vector(evidence=evidence).shape[1] == vocabulary
