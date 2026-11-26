"""Training the transaction model like the receipt models: forward-chaining
folds by date (EV-2), the most-frequent baseline (EV-3), one state file
per dataset configuration (DS-2), resumable progress."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from hledger_ai.evaluation.folds import forward_chaining_folds
from hledger_ai.models.catalogue import ModelSpec
from hledger_ai.training.state import ModelState, StateStore, StrategyState
from hledger_ai.transactions.dataset import TransactionData
from hledger_ai.transactions.model import (
    ENGINE_VERSION,
    MostFrequent,
    NotEnoughData,
    TransactionModel,
    score,
)
from typeguard import typechecked

STRATEGY_ID = "ST-21"
FIELD = "transaction_category"
MIN_EXAMPLES = 2


@typechecked
def train_transaction_model(
    *,
    spec: ModelSpec,
    data: TransactionData,
    store: StateStore,
    folds: int = 4,
    force: bool = False,
    on_progress: Optional[Callable[[ModelState], None]] = None,
) -> ModelState:
    """Train *spec* on *data*; returns the state (also saved to disk)."""
    if len(data.examples) < MIN_EXAMPLES:
        raise NotEnoughData(
            f"{spec.name}: {len(data.examples)} categorised transaction(s) is"
            " not enough to learn from; add rules first"
            " (hledger_preprocessor --categorise-transactions)"
        )
    hyperparameters = {"folds": folds, "strategies": [STRATEGY_ID]}
    state = store.load(model=spec.name, dataset_hash=data.dataset_hash)
    if (
        state is not None
        and not force
        and state.is_current(
            dataset_hash=data.dataset_hash,
            model_version=ENGINE_VERSION,
            hyperparameters=hyperparameters,
        )
    ):
        return state
    resumable = (
        state is not None
        and not force
        and not state.finished
        and state.model_version == ENGINE_VERSION
        and state.hyperparameters == hyperparameters
        and STRATEGY_ID in state.strategies
    )
    if not resumable:
        state = ModelState(
            model=spec.name,
            dataset_hash=data.dataset_hash,
            private=data.private,
            model_version=ENGINE_VERSION,
            hyperparameters=hyperparameters,
        )
    assert state is not None

    def report() -> None:
        store.save(state=state)
        if on_progress is not None:
            on_progress(state)

    examples = sorted(data.examples, key=lambda e: e.row.date)
    splits = forward_chaining_folds(count=len(examples), folds=folds)
    state.fits_total = len(splits) + 1
    progress = state.strategies.setdefault(
        STRATEGY_ID,
        StrategyState(infer_total=sum(len(test) for _, test in splits)),
    )
    report()

    for index, (train_range, test_range) in enumerate(splits):
        if str(index) in progress.folds:
            continue  # scored before the run was interrupted
        train = [examples[i] for i in train_range]
        test = [examples[i] for i in test_range]
        started = time.monotonic()
        try:
            model = TransactionModel().fit(examples=train)
        except NotEnoughData:
            model = None
        fit_seconds = time.monotonic() - started
        started = time.monotonic()
        if model is None:
            scored = {"correct": 0, "total": len(test), "baseline_correct": 0}
        else:
            scored = score(
                model=model,
                baseline=MostFrequent().fit(examples=train),
                test=test,
            )
        progress.folds[str(index)] = {FIELD: scored}
        progress.infer_done += len(test)
        progress.infer_seconds += time.monotonic() - started
        state.fits_done += 1
        state.fit_seconds += fit_seconds
        report()

    started = time.monotonic()
    path = store.engine_path(model=spec.name, dataset_hash=data.dataset_hash)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    TransactionModel().fit(examples=examples).save(path=path)
    state.fit_seconds += time.monotonic() - started
    state.fits_done = state.fits_total
    state.finished = True
    report()
    return state
