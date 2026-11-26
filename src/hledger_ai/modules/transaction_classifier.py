"""Classify bank CSV transactions into categories with the locally trained
``csv-linear`` model (hledger_ai.transactions).  Implements the
``TransactionCategoryModel`` protocol of hledger-core: the result goes into
the AI column of the preprocessed CSVs next to the rule-based one."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from hledger_ai.transactions.model import TransactionModel
from typeguard import typechecked

from hledger_core.categorisation.Categories import CategoryNamespace
from hledger_core.categorisation.rules import RowView
from hledger_core.generics.Transaction import Transaction

log = logging.getLogger(__name__)


class TransactionClassifier:
    # The preprocessed CSVs carry this column name: keep it.
    name = "TransactionClassifier"

    @typechecked
    def __init__(
        self,
        *,
        model: Optional[TransactionModel] = None,
        config: Optional[Any] = None,
    ) -> None:
        self._model = model
        self._config = config
        self._loaded = model is not None

    @staticmethod
    @typechecked
    def from_config(config: Any) -> "TransactionClassifier":
        """With the newest trained model of the root in *config* (or of
        ``ai.transaction_model_root``), loaded on first use: a check that
        never classifies never pays for the model.  Without a model it
        answers ''."""
        return TransactionClassifier(config=config)

    def _load(self) -> Optional[TransactionModel]:
        if not self._loaded:
            self._loaded = True
            from hledger_ai.transactions.suggest import load_model_for_config

            try:
                self._model = load_model_for_config(config=self._config)
            except Exception:  # a broken model never stops the pipeline
                log.warning("transaction model not loadable", exc_info=True)
                self._model = None
        return self._model

    @property
    def is_available(self) -> bool:
        return self._load() is not None

    def suggest(
        self, transaction: Transaction, *, limit: int = 5
    ) -> List[Tuple[str, float]]:
        model = self._load()
        if model is None:
            return []
        return model.suggest(
            row=RowView.from_transaction(transaction), limit=limit
        )

    def classify(
        self,
        transaction: Transaction,
        category_namespace: CategoryNamespace,
    ) -> str:
        """The most likely category path, or '' without a model."""
        del category_namespace  # the model's classes come from the rules
        ranked = self.suggest(transaction, limit=1)
        return ranked[0][0] if ranked else ""
