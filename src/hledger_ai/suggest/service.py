"""Suggestions from everything that is trained (CI-1 to CI-4, EV-4).

Loads the newest finished training of every model and answers each field
with the model and strategy that scored best on unseen receipts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.dataset.configuration import DatasetRegistry
from hledger_ai.models.catalogue import MODELS, ModelSpec
from hledger_ai.paths import AiPaths
from hledger_ai.receipt_fields import FIELDS
from hledger_ai.selection import read_selection
from hledger_ai.strategies import get_strategy
from hledger_ai.suggest.engine import (
    Suggestion,
    SuggestionEngine,
    image_strategy_id,
)
from hledger_ai.suggest.evidence import ReceiptEvidence
from hledger_ai.suggest.policy import ModelView, choose
from hledger_ai.training.state import ModelState, StateStore

log = logging.getLogger(__name__)

# SP-2: time between confirming a field and reaching the one it improves,
# until the labeller's own timings say otherwise (SP-5).
DEFAULT_BUDGET_SECONDS = 23.0


@typechecked
@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    state: ModelState
    engine: SuggestionEngine


@typechecked
class SuggestionService:
    def __init__(
        self,
        *,
        models: List[LoadedModel],
        budget_seconds: Optional[Callable[[str], float]] = None,
    ) -> None:
        self._models = models
        self._budget = budget_seconds or (lambda _: DEFAULT_BUDGET_SECONDS)

    @staticmethod
    def load(
        *,
        ai: AiPaths,
        budget_seconds: Optional[Callable[[str], float]] = None,
    ) -> "SuggestionService":
        """The newest finished training of every model trained here."""
        store = StateStore(models_dir=ai.models_dir)
        newest_first = list(
            reversed(DatasetRegistry(path=ai.registry_path).configurations())
        )
        chosen = read_selection(ai=ai)  # HW-5
        models: List[LoadedModel] = []
        for spec in MODELS:
            if not spec.trained_here:
                continue
            if chosen is not None and spec.name not in chosen:
                continue
            finished = []
            for configuration in newest_first:
                state = store.load(
                    model=spec.name, dataset_hash=configuration.dataset_hash
                )
                path = store.engine_path(
                    model=spec.name, dataset_hash=configuration.dataset_hash
                )
                if (
                    state is not None
                    and state.finished
                    and os.path.isfile(path)
                ):
                    finished.append((state, path))
            # Models trained on the user's own (private) labels come first;
            # public-only training is used when nothing else exists.
            finished.sort(key=lambda pair: not pair[0].private)
            if finished:
                state, path = finished[0]
                models.append(
                    LoadedModel(
                        spec=spec,
                        state=state,
                        engine=SuggestionEngine.load(path=path),
                    )
                )
        return SuggestionService(models=models, budget_seconds=budget_seconds)

    def ocr_engine_ids(self) -> List[str]:
        """OCR engines whose text the loaded models need."""
        return sorted(
            {m.spec.ocr_engine for m in self._models if m.spec.ocr_engine}
        )

    def dataset_hashes(self) -> Dict[str, str]:
        return {m.spec.name: m.state.dataset_hash for m in self._models}

    def is_empty(self) -> bool:
        return not self._models

    def suggest(
        self,
        *,
        evidence: Dict[str, ReceiptEvidence],
        known: Dict[str, Any],
        limit: int = 5,
    ) -> Dict[str, List[Suggestion]]:
        """Ranked suggestions for every field not in *known*.

        *evidence* maps an OCR engine id to what that engine read.
        """
        usable = [
            m
            for m in self._models
            if not m.spec.uses_image or m.spec.ocr_engine in evidence
        ]
        views = [
            ModelView(
                name=m.spec.name,
                tier=m.spec.tier,
                uses_image=m.spec.uses_image,
                state=m.state,
            )
            for m in usable
        ]
        by_name = {m.spec.name: m for m in usable}
        result: Dict[str, List[Suggestion]] = {}
        for field in FIELDS:
            if field in known:
                continue  # CI-4
            choice = choose(
                field=field,
                known=set(known),
                has_image=True,
                models=views,
                budget_seconds=self._budget,
            )
            if choice is None:
                continue
            model = by_name[choice.model]
            strategy = get_strategy(strategy_id=choice.strategy_id)
            active = {strategy.id}
            if model.spec.uses_image:
                active.add(image_strategy_id(field=field) or "")
            answer = model.engine.suggest(
                evidence=(
                    evidence[model.spec.ocr_engine]
                    if model.spec.uses_image
                    else None
                ),
                known={k: known[k] for k in strategy.known if k in known},
                strategies=active,
                fields={field},
                limit=limit,
            )
            if field in answer:
                result[field] = answer[field]
        return result
