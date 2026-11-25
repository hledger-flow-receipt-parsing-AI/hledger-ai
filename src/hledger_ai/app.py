"""Use cases behind the command line and the labeller: gather the dataset,
train what is named or selected, report."""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from typeguard import typechecked

from hledger_ai.dataset.configuration import DatasetRegistry, dataset_hash
from hledger_ai.dataset.examples import (
    ReceiptExample,
    load_private_examples,
    load_public_examples,
)
from hledger_ai.hardware import (
    Device,
    benchmark_model,
    probe_device,
    select_models,
)
from hledger_ai.models.catalogue import (
    MODELS,
    ModelSpec,
    build_ocr_engine,
    get_model,
)
from hledger_ai.ocr.cache import OcrCache
from hledger_ai.ocr.types import OcrEngine
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.privacy import make_private
from hledger_ai.selection import read_selection
from hledger_ai.strategies import STRATEGIES
from hledger_ai.training.state import ModelState, StateStore
from hledger_ai.training.trainer import TrainingData, train_model

log = logging.getLogger(__name__)

# TR-4: a receipt whose suggestion the user had to correct counts double.
CORRECTED_WEIGHT = 2.0


class UnknownName(ValueError):
    """A name given to ``--train`` is neither a model nor a strategy."""


@typechecked
def collect_training_data(
    *, finance: FinancePaths, public_only: bool
) -> TrainingData:
    """PR-2: the public dataset alone, or the public and private receipts."""
    examples: List[ReceiptExample] = load_public_examples(
        training_dataset_dir=finance.training_dataset_dir,
        crop_suffix=finance.crop_suffix,
    )
    if not public_only:
        examples += load_private_examples(
            labels_dir=finance.labels_dir,
            processed_dir=finance.processed_dir,
            label_filename=finance.label_filename,
            crop_suffix=finance.crop_suffix,
            crop_ext=finance.crop_ext,
        )
    private = any(example.private for example in examples)
    return TrainingData(
        examples=tuple(examples),
        private=private,
        dataset_hash=dataset_hash(
            labels=[e.label or {} for e in examples], private=private
        ),
    )


@typechecked
def resolve_names(
    *, names: Sequence[str]
) -> Tuple[List[str], Optional[Set[str]]]:
    """TR-3: split ``--train`` names into model names and strategy ids
    (None: all strategies)."""
    model_names = [spec.name for spec in MODELS]
    strategy_ids = [strategy.id for strategy in STRATEGIES]
    unknown = [n for n in names if n not in model_names + strategy_ids]
    if unknown:
        raise UnknownName(
            f"Unknown name(s): {', '.join(unknown)}.\n"
            f"Models: {', '.join(model_names)}.\n"
            f"Strategies: {', '.join(strategy_ids)}."
        )
    models = [n for n in names if n in model_names]
    strategies = {n for n in names if n in strategy_ids}
    return models, (strategies or None)


@typechecked
def example_weights(
    *, data: TrainingData, corrections_path: str
) -> Dict[str, float]:
    """TR-4: receipts with a logged correction get a higher weight."""
    if not os.path.isfile(corrections_path):
        return {}
    corrected: Set[str] = set()
    with open(corrections_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                path = json.loads(line).get("image_path")
            except json.JSONDecodeError:
                continue
            if path:
                corrected.add(path)
    return {
        example.receipt_id: CORRECTED_WEIGHT
        for example in data.examples
        if corrected & set(example.facts.image_paths)
    }


@typechecked
def benchmarks(
    *,
    ai: AiPaths,
    specs: Sequence[ModelSpec],
    announce: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """HW-3 benchmark per model, measured once and remembered."""
    path = os.path.join(ai.root, "hardware", "benchmarks.json")
    known: Dict[str, float] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            known = json.load(handle)
    changed = False
    for spec in specs:
        if spec.name in known or spec.ocr_engine is None:
            continue
        if announce is not None:
            announce(f"measuring how fast {spec.name} reads a photo here ...")
        try:
            seconds = benchmark_model(spec=spec)
        except Exception:  # a broken runtime must not stop the others
            log.exception("Benchmark of %s failed", spec.name)
            seconds = None
        if seconds is not None:
            known[spec.name] = seconds
            changed = True
    if changed:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(known, handle, indent=1)
    return known


@typechecked
def default_models(
    *,
    ai: AiPaths,
    device: Optional[Device] = None,
    ollama_url: str = "",
    announce: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """HW-4: the trainable models that run flawlessly on this device, or
    the user's own choice when one was made (HW-5)."""
    chosen = read_selection(ai=ai)
    if chosen is not None:
        return chosen
    device = device or probe_device(
        storage_dir=ai.root,
        ollama_url=ollama_url or "http://localhost:11434",
        qvac_dir=ai.qvac_dir,
    )
    candidates = [
        spec
        for spec in MODELS
        if spec.trained_here and spec.runtime in device.runtimes
    ]
    return select_models(
        specs=candidates,
        device=device,
        benchmarks=benchmarks(ai=ai, specs=candidates, announce=announce),
        accuracy={},
    )


@typechecked
def train(
    *,
    finance: FinancePaths,
    ai: AiPaths,
    names: Sequence[str] = (),
    public_only: bool = False,
    force: bool = False,
    folds: int = 4,
    ocr_engines: Optional[Dict[str, OcrEngine]] = None,
    on_progress: Optional[Callable[[ModelState], None]] = None,
) -> List[ModelState]:
    """TR-2: train every selected model that is out of date.

    *names* limits the run to those models and strategies (TR-3);
    *ocr_engines* replaces the real OCR engines (tests).
    """
    models, strategy_ids = resolve_names(names=names)
    make_private(folder=ai.root)
    data = collect_training_data(finance=finance, public_only=public_only)
    DatasetRegistry(path=ai.registry_path).register(
        dataset_hash=data.dataset_hash,
        private=data.private,
        receipt_count=len(data.examples),
    )
    store = StateStore(models_dir=ai.models_dir)
    cache = OcrCache(cache_dir=ai.ocr_cache_dir)
    weights = example_weights(data=data, corrections_path=ai.corrections_path)
    states: List[ModelState] = []
    for name in models or default_models(ai=ai):
        spec = get_model(name=name)
        if not spec.trained_here:
            raise UnknownName(
                f"{name} is not trained by --train ({spec.note})."
            )
        engine: Optional[OcrEngine] = None
        if spec.ocr_engine is not None:
            engine = (ocr_engines or {}).get(
                spec.ocr_engine
            ) or build_ocr_engine(engine_id=spec.ocr_engine)
        states.append(
            train_model(
                spec=spec,
                data=data,
                store=store,
                cache=cache,
                ocr_engine=engine,
                strategy_ids=strategy_ids,
                folds=folds,
                force=force,
                weights=weights,
                on_progress=on_progress,
            )
        )
    return states
