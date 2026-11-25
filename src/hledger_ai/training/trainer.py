"""Train one model on one dataset configuration (TR-2, TR-3).

Per time-ordered fold (EV-2) the model is fitted on the older receipts and
every strategy is scored on the newer ones; then it is fitted on all
receipts and saved. Progress is written after every step, so an interrupted
run continues where it stopped (BG-1) and a finished, unchanged
configuration is never trained twice (DS-2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from typeguard import typechecked

from hledger_ai.candidates.dates import capture_time
from hledger_ai.dataset.examples import ReceiptExample
from hledger_ai.evaluation.folds import forward_chaining_folds
from hledger_ai.evaluation.scoring import (
    Baseline,
    needed_fields,
    scorable_strategies,
    score_engine,
)
from hledger_ai.models.catalogue import ModelSpec
from hledger_ai.models.experts import TrainingExample
from hledger_ai.ocr.cache import OcrCache, file_sha256
from hledger_ai.ocr.types import OcrEngine, OcrResult
from hledger_ai.suggest.engine import ENGINE_VERSION, SuggestionEngine
from hledger_ai.suggest.evidence import ReceiptEvidence
from hledger_ai.training.state import ModelState, StateStore, StrategyState

log = logging.getLogger(__name__)

_SAVE_EVERY_PHOTOS = 10


@typechecked
@dataclass(frozen=True)
class TrainingData:
    """One dataset configuration: the examples and their identity."""

    examples: Tuple[ReceiptExample, ...]
    private: bool  # DS-5 / PR-2
    dataset_hash: str  # DS-1


@typechecked
def _read_photos(
    *,
    examples: Sequence[ReceiptExample],
    cache: OcrCache,
    ocr_engine: OcrEngine,
    state: ModelState,
    report: Callable[[], None],
) -> Dict[str, Tuple[OcrResult, ...]]:
    """OCR of every photo, cached ones first so the ETA is honest."""
    photos = [(e.receipt_id, path) for e in examples for path in e.image_paths]
    state.ocr_total = len(photos)
    state.ocr_done = 0
    found: Dict[Tuple[str, str], OcrResult] = {}
    to_read: List[Tuple[str, str, str]] = []
    for receipt_id, path in photos:
        sha = file_sha256(path=path)
        cached = cache.get(engine=ocr_engine.engine_id, image_sha256=sha)
        if cached is None:
            to_read.append((receipt_id, path, sha))
        else:
            found[(receipt_id, path)] = cached
            state.ocr_done += 1
    report()
    for count, (receipt_id, path, sha) in enumerate(to_read, start=1):
        started = time.monotonic()
        result = ocr_engine.read(image_path=path, image_sha256=sha)
        cache.put(result=result)
        found[(receipt_id, path)] = result
        state.ocr_seconds += time.monotonic() - started
        state.ocr_read += 1
        state.ocr_done += 1
        if count % _SAVE_EVERY_PHOTOS == 0 or count == len(to_read):
            report()
    return {
        e.receipt_id: tuple(found[(e.receipt_id, p)] for p in e.image_paths)
        for e in examples
    }


@typechecked
def train_model(
    *,
    spec: ModelSpec,
    data: TrainingData,
    store: StateStore,
    cache: OcrCache,
    ocr_engine: Optional[OcrEngine],
    strategy_ids: Optional[Set[str]] = None,
    folds: int = 4,
    force: bool = False,
    weights: Optional[Dict[str, float]] = None,
    on_progress: Optional[Callable[[ModelState], None]] = None,
) -> ModelState:
    """Train *spec* on *data*; returns the state (also saved to disk).

    *strategy_ids* limits training to those strategies (TR-3); *weights*
    maps receipt ids to example weights (TR-4).
    """
    wanted = scorable_strategies(
        use_image=spec.uses_image, strategy_ids=strategy_ids
    )
    hyperparameters = {
        "folds": folds,
        "ocr_engine": spec.ocr_engine,
        "strategies": sorted(s.id for s in wanted),
    }
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
        log.info("%s is up to date for %s", spec.name, data.dataset_hash[:12])
        return state
    resumable = (
        state is not None
        and not force
        and not state.finished
        and state.model_version == ENGINE_VERSION
        and state.hyperparameters == hyperparameters
    )
    if not resumable:
        state = ModelState(
            model=spec.name,
            dataset_hash=data.dataset_hash,
            private=data.private,
            model_version=ENGINE_VERSION,
            hyperparameters=hyperparameters,
        )

    def report() -> None:
        store.save(state=state)
        if on_progress is not None:
            on_progress(state)

    ocr: Dict[str, Tuple[OcrResult, ...]] = {}
    if spec.uses_image:
        if ocr_engine is None:
            raise ValueError(f"{spec.name} needs an OCR engine")
        ocr = _read_photos(
            examples=data.examples,
            cache=cache,
            ocr_engine=ocr_engine,
            state=state,
            report=report,
        )
    ordered = sorted(
        data.examples, key=lambda e: e.facts.the_datetime or datetime.min
    )
    examples = [
        TrainingExample(
            facts=e.facts,
            evidence=ReceiptEvidence(
                ocr=ocr.get(e.receipt_id, ()),
                captured_at=capture_time(
                    image_paths=list(e.facts.image_paths) + list(e.image_paths)
                ),
            ),
            weight=(weights or {}).get(e.receipt_id, 1.0),
        )
        for e in ordered
    ]
    fields = needed_fields(strategies=wanted)
    splits = forward_chaining_folds(count=len(examples), folds=folds)
    state.fits_total = len(splits) + 1
    held_out = sum(len(test) for _, test in splits)
    for strategy in wanted:
        state.strategies.setdefault(strategy.id, StrategyState())
        state.strategies[strategy.id].infer_total = held_out

    for index, (train_range, test_range) in enumerate(splits):
        if all(str(index) in state.strategies[s.id].folds for s in wanted):
            continue  # scored before the run was interrupted
        train = [examples[i] for i in train_range]
        test = [examples[i] for i in test_range]
        started = time.monotonic()
        engine = SuggestionEngine.train(
            examples=train, use_image=spec.uses_image, fields=fields
        )
        fit_seconds = time.monotonic() - started
        scores = score_engine(
            engine=engine,
            baseline=Baseline.fit(examples=train),
            test=test,
            use_image=spec.uses_image,
            strategy_ids={s.id for s in wanted},
        )
        for strategy in wanted:
            scored = scores.get(strategy.id, {})
            progress = state.strategies[strategy.id]
            progress.folds[str(index)] = {
                name: {
                    "correct": score.correct,
                    "total": score.total,
                    "baseline_correct": score.baseline_correct,
                }
                for name, score in scored.items()
            }
            progress.infer_done += len(test)
            progress.infer_seconds += max(
                (score.seconds for score in scored.values()), default=0.0
            )
        state.fits_done += 1
        state.fit_seconds += fit_seconds
        report()

    started = time.monotonic()
    SuggestionEngine.train(
        examples=examples, use_image=spec.uses_image, fields=fields
    ).save(
        path=_prepared(
            path=store.engine_path(
                model=spec.name, dataset_hash=data.dataset_hash
            )
        )
    )
    state.fit_seconds += time.monotonic() - started
    state.fits_done = state.fits_total
    state.finished = True
    report()
    return state


@typechecked
def _prepared(*, path: str) -> str:
    import os

    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    return path
