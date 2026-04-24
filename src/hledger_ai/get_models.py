"""Model registry: assembles modular AI extraction pipeline and classifiers.

This replaces the old monolithic DonutAI + ExampleAIModel setup with
the modular extraction pipeline (independent modules + cross-checks)
and the SetFit/Qwen3 transaction classifier.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.cross_checks.category_from_shop import (
    CategoryFromShop,
)
from hledger_ai.cross_checks.date_sanity import DateSanityCheck
from hledger_ai.cross_checks.shop_address_verifier import (
    ShopAddressVerifier,
)
from hledger_ai.cross_checks.total_items_consistency import (
    TotalItemsConsistency,
)
from hledger_ai.modules.category import CategoryModule
from hledger_ai.modules.date_time import DateTimeModule
from hledger_ai.modules.line_items import LineItemsModule
from hledger_ai.modules.shop_identifier import ShopIdentifierModule
from hledger_ai.modules.totals import TotalsModule
from hledger_ai.modules.transaction_classifier import (
    TransactionClassifier,
)
from hledger_ai.ollama_client import OllamaClient
from hledger_ai.pipeline import ExtractionPipeline
from hledger_core.categorisation.Categories import CategoryNamespace
from hledger_core.generics.enums import ClassifierType, LogicType
from hledger_core.generics.TransactionCategoryModel import (
    TransactionCategoryModel,
)

log = logging.getLogger(__name__)


def _get_rule_based_model():
    """Deferred import of the rule-based model from the orchestrator."""
    from hledger_preprocessor.categorisation.rule_based.rule_based_eg0 import (
        ExampleRuleBasedModel,
    )

    return ExampleRuleBasedModel()


@typechecked
def build_extraction_pipeline(
    *,
    ollama_url: str = "http://localhost:11434",
    vlm_model: str = "qwen3-vl:2b",
    known_shop_categories: Optional[Dict[str, str]] = None,
    category_tree: Optional[List[str]] = None,
    known_locations: Optional[
        Dict[str, List[Dict[str, str]]]
    ] = None,
    shop_category_counts: Optional[
        Dict[str, Dict[str, int]]
    ] = None,
    text_model: str = "qwen3:0.6b",
) -> ExtractionPipeline:
    """Build the full modular extraction pipeline.

    Returns an ``ExtractionPipeline`` with all VLM modules,
    category module, and cross-checks wired together.
    """
    client = OllamaClient(base_url=ollama_url)

    modules = [
        ShopIdentifierModule(client=client, model=vlm_model),
        LineItemsModule(client=client, model=vlm_model),
        TotalsModule(client=client, model=vlm_model),
        DateTimeModule(client=client, model=vlm_model),
        CategoryModule(
            known_mappings=known_shop_categories,
            category_tree=category_tree,
            client=client,
            model=text_model,
        ),
    ]

    cross_checks = [
        TotalItemsConsistency(),
        ShopAddressVerifier(known_locations=known_locations),
        CategoryFromShop(
            shop_category_counts=shop_category_counts,
        ),
        DateSanityCheck(),
    ]

    return ExtractionPipeline(
        modules=modules, cross_checks=cross_checks
    )


@typechecked
def build_transaction_classifier(
    *,
    ollama_url: str = "http://localhost:11434",
    setfit_model_dir: Optional[str] = None,
    category_namespace: Optional[CategoryNamespace] = None,
) -> TransactionClassifier:
    """Build the transaction classifier with SetFit + zero-shot cascade."""
    client = OllamaClient(base_url=ollama_url)
    return TransactionClassifier(
        client=client,
        setfit_model_dir=setfit_model_dir,
        category_namespace=category_namespace,
    )


def get_models(
    *, quick_categorisation: bool
) -> Dict[ClassifierType, Dict[LogicType, Any]]:
    """Build the model registry.

    Backward-compatible entry point. For new code, prefer
    ``build_extraction_pipeline()`` and
    ``build_transaction_classifier()`` directly.
    """
    classifier = TransactionClassifier()

    if quick_categorisation:
        classifiers: Dict[ClassifierType, Dict[LogicType, Any]] = {
            ClassifierType.TRANSACTION_CATEGORY: (
                get_transaction_classification_models(
                    classifier=classifier,
                )
            ),
        }
    else:
        classifiers = {
            ClassifierType.TRANSACTION_CATEGORY: (
                get_transaction_classification_models(
                    classifier=classifier,
                )
            ),
        }
    return classifiers


@typechecked
def get_transaction_classification_models(
    *,
    classifier: Optional[TransactionClassifier] = None,
) -> Dict[LogicType, List[TransactionCategoryModel]]:
    """Get transaction classification models."""
    ai_model: TransactionCategoryModel = (
        classifier if classifier is not None
        else TransactionClassifier()
    )
    rule_based_model: TransactionCategoryModel = (
        _get_rule_based_model()
    )
    return {
        LogicType.AI: [ai_model],
        LogicType.RULE_BASED: [rule_based_model],
    }
