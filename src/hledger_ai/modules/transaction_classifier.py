"""Classify bank CSV transactions into spending categories."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.ollama_client import OllamaClient
from hledger_core.categorisation.Categories import CategoryNamespace
from hledger_core.generics.Transaction import Transaction

log = logging.getLogger(__name__)


def _flatten_hierarchy(
    node: Dict[str, Any], prefix: str = ""
) -> List[str]:
    """Recursively flatten a category hierarchy dict to leaf paths."""
    paths: List[str] = []
    for key, children in node.items():
        path = f"{prefix}:{key}" if prefix else key
        if isinstance(children, dict) and children:
            paths.extend(_flatten_hierarchy(children, path))
        else:
            paths.append(path)
    return paths


@typechecked
class TransactionClassifier:
    """Classifies bank transactions into spending categories.

    Classification cascade:
    1. SetFit model (if trained and available) — instant, accurate.
    2. Zero-shot Qwen3-0.6B via Ollama — slower, reasonable.
    3. Returns ``None`` if nothing works.

    Implements the ``TransactionCategoryModel`` protocol from
    ``hledger-core`` via the ``classify`` method.
    """

    ZERO_SHOT_MODEL = "qwen3:0.6b"
    name = "TransactionClassifier"

    def __init__(
        self,
        *,
        client: Optional[OllamaClient] = None,
        setfit_model_dir: Optional[str] = None,
        category_namespace: Optional[CategoryNamespace] = None,
    ) -> None:
        self._client = client
        self._setfit_model = None
        self._setfit_dir = setfit_model_dir
        self._namespace = category_namespace
        self._categories: List[str] = []

        if category_namespace is not None:
            self._categories = _flatten_hierarchy(
                category_namespace._hierarchy
            )

        if setfit_model_dir is not None:
            self._try_load_setfit(setfit_model_dir)

    def _try_load_setfit(self, model_dir: str) -> None:
        """Attempt to load a trained SetFit model."""
        model_path = Path(model_dir)
        if not model_path.exists():
            log.info(
                "No SetFit model at %s, using zero-shot",
                model_dir,
            )
            return
        try:
            from setfit import SetFitModel

            self._setfit_model = SetFitModel.from_pretrained(
                str(model_path)
            )
            log.info("Loaded SetFit model from %s", model_dir)
        except ImportError:
            log.warning(
                "setfit not installed; "
                "install with: pip install hledger-ai[setfit]"
            )
        except Exception:
            log.exception(
                "Failed to load SetFit model from %s",
                model_dir,
            )

    def classify(
        self,
        transaction: Transaction,
        category_namespace: CategoryNamespace,
    ) -> str:
        """Classify a single transaction. Protocol-compatible.

        Returns a colon-separated category path (e.g.
        ``"groceries:ekoplaza"``), or empty string on failure.
        """
        text = self._transaction_to_text(transaction)

        # Update categories if namespace changed
        if (
            category_namespace is not self._namespace
            and category_namespace is not None
        ):
            self._namespace = category_namespace
            self._categories = _flatten_hierarchy(
                category_namespace._hierarchy
            )

        # Tier 1: SetFit
        if self._setfit_model is not None:
            result = self._classify_setfit(text)
            if result:
                return result

        # Tier 2: Zero-shot LLM
        if self._client is not None and self._categories:
            result = self._classify_zero_shot(text)
            if result:
                return result

        return ""

    def classify_batch(
        self,
        transactions: List[Transaction],
        category_namespace: CategoryNamespace,
    ) -> List[str]:
        """Classify a batch of transactions."""
        return [
            self.classify(t, category_namespace)
            for t in transactions
        ]

    def _transaction_to_text(
        self, transaction: Transaction
    ) -> str:
        """Convert a transaction to classifier input text."""
        parts: List[str] = []
        desc = getattr(transaction, "description", None)
        if desc:
            parts.append(str(desc))
        other = getattr(transaction, "other_party_name", None)
        if other:
            parts.append(str(other))
        return " | ".join(parts) if parts else "unknown"

    def _classify_setfit(self, text: str) -> Optional[str]:
        """Classify using the trained SetFit model."""
        try:
            predictions = self._setfit_model.predict([text])
            label = predictions[0]
            if isinstance(label, str) and label:
                log.debug("SetFit: '%s' -> '%s'", text, label)
                return label
        except Exception:
            log.exception("SetFit classification failed")
        return None

    def _classify_zero_shot(self, text: str) -> Optional[str]:
        """Classify using Qwen3-0.6B zero-shot."""
        assert self._client is not None

        cat_list = "\n".join(
            f"- {c}" for c in self._categories
        )
        prompt = (
            "Classify this bank transaction into exactly "
            "one of the following categories:\n"
            f"{cat_list}\n\n"
            f"Transaction: {text}\n\n"
            "Reply with only the category path, nothing else."
        )

        schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
            },
            "required": ["category"],
        }

        try:
            result = self._client.generate_structured(
                model=self.ZERO_SHOT_MODEL,
                prompt=prompt,
                schema=schema,
            )
            category = result.get("category", "")
            if category in self._categories:
                log.debug(
                    "Zero-shot: '%s' -> '%s'", text, category
                )
                return category
            log.warning(
                "Zero-shot returned unknown category: '%s'",
                category,
            )
            return category if category else None
        except Exception:
            log.exception("Zero-shot classification failed")
            return None
