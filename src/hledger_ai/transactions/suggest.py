"""Using the trained transaction model: suggestions for a row or a group,
and rule drafts for every uncategorised group of a CSV."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from hledger_ai.dataset.configuration import DatasetRegistry
from hledger_ai.paths import AiPaths
from hledger_ai.training.state import ModelState, StateStore
from hledger_ai.transactions.model import TransactionModel
from typeguard import typechecked

from hledger_core.categorisation.rules import (
    Draft,
    Group,
    ProbeReport,
    RowView,
    RuleStore,
    Suggestion,
    group_rows,
    probe,
)

MODEL_NAME = "csv-linear"
GroupSuggester = Callable[[Group], List[Suggestion]]


@typechecked
def latest_engine(*, ai: AiPaths) -> Optional[Tuple[str, ModelState]]:
    """The newest finished transaction model of this AI folder."""
    store = StateStore(models_dir=ai.models_dir)
    configurations = DatasetRegistry(path=ai.registry_path).configurations()
    for configuration in reversed(configurations):
        if configuration.kind != "transactions":
            continue
        state = store.load(
            model=MODEL_NAME, dataset_hash=configuration.dataset_hash
        )
        path = store.engine_path(
            model=MODEL_NAME, dataset_hash=configuration.dataset_hash
        )
        if state is not None and state.finished and os.path.isfile(path):
            return path, state
    return None


@typechecked
def model_paths(*, config: Any) -> AiPaths:
    """This root's AI folder, or the one of ``ai.transaction_model_root``."""
    ai_config = getattr(config, "ai", None)
    other = (
        getattr(ai_config, "transaction_model_root", None)
        if ai_config
        else None
    )
    if other:
        return AiPaths.for_root(finance_root=other)
    return AiPaths.from_config(config=config)


@typechecked
def load_model(*, ai: AiPaths) -> Optional[TransactionModel]:
    found = latest_engine(ai=ai)
    if found is None:
        return None
    return TransactionModel.load(path=found[0])


@typechecked
def load_model_for_config(*, config: Any) -> Optional[TransactionModel]:
    return load_model(ai=model_paths(config=config))


@typechecked
def load_group_suggester(*, config: Any) -> Optional[GroupSuggester]:
    """What the categorisation TUI asks for: ranked categories per group,
    None when no model is trained yet."""
    model = load_model_for_config(config=config)
    if model is None:
        return None
    return lambda group: model.suggest_group(rows=list(group.rows))


@typechecked
@dataclass(frozen=True)
class RuleDraft:
    """A rule the model proposes for one group of uncategorised rows."""

    group: Group
    draft: Draft
    suggestions: List[Suggestion]
    probe: ProbeReport

    @property
    def confidence(self) -> float:
        return self.suggestions[0][1] if self.suggestions else 0.0

    @property
    def is_one_hot(self) -> bool:
        return self.probe.is_one_hot


@typechecked
def draft_rules(
    *,
    model: Optional[TransactionModel],
    store: RuleStore,
    rows: Sequence[RowView],
    uncategorised: Sequence[RowView],
    limit: int = 5,
) -> List[RuleDraft]:
    """One draft per group of *uncategorised* rows: the group's condition,
    the model's categories (best first), and what the draft does to every
    row of the root.  Without a model the drafts carry no category."""
    drafts: List[RuleDraft] = []
    for group in group_rows(uncategorised):
        suggestions = (
            model.suggest_group(rows=list(group.rows), limit=limit)
            if model is not None
            else []
        )
        draft = Draft(
            direction=group.direction,
            match=[group.draft_condition()],
            target=suggestions[0][0] if suggestions else "",
        )
        drafts.append(
            RuleDraft(
                group=group,
                draft=draft,
                suggestions=suggestions,
                probe=probe(store=store, draft=draft.rule(), rows=rows),
            )
        )
    return drafts
