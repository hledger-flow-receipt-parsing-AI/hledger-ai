"""Convert extraction pipeline output to AISuggestion objects for the TUI."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.pipeline import ExtractionPipeline
from hledger_core.AISuggestion import AISuggestion

log = logging.getLogger(__name__)


@typechecked
class AIReceiptSuggester:
    """Runs the extraction pipeline and converts output to AISuggestion lists.

    Returns a dict keyed by question identifier, each value a list of
    ``AISuggestion`` objects sorted by descending probability.
    """

    def __init__(
        self, *, pipeline: ExtractionPipeline
    ) -> None:
        self._pipeline = pipeline

    def suggest(
        self, *, image_path: str
    ) -> Dict[str, List[AISuggestion]]:
        """Run all modules and convert to TUI suggestions."""
        try:
            result = self._pipeline.run(image_path=image_path)
        except Exception:
            log.exception(
                "Extraction pipeline failed for %s", image_path
            )
            return {}
        return self._convert(result)

    def _convert(
        self, result: Dict[str, Any]
    ) -> Dict[str, List[AISuggestion]]:
        suggestions: Dict[str, List[AISuggestion]] = {}

        # Date
        the_date = result.get("the_date")
        if the_date is not None:
            suggestions["receipt_date"] = [
                AISuggestion(
                    question=str(the_date),
                    probability=0.85,
                    model_name="date_time",
                )
            ]

        # Shop name
        shop = result.get("shop_identifier")
        if isinstance(shop, dict):
            name = shop.get("name")
            if name:
                suggestions["shop_name"] = [
                    AISuggestion(
                        question=str(name),
                        probability=0.85,
                        model_name="shop_identifier",
                    )
                ]

            # Shop address
            address = shop.get("address")
            if isinstance(address, dict):
                addr_parts = []
                for k in ("street", "house_nr", "zipcode", "city"):
                    val = address.get(k)
                    if val:
                        addr_parts.append(str(val))
                if addr_parts:
                    suggestions["shop_address"] = [
                        AISuggestion(
                            question=", ".join(addr_parts),
                            probability=0.80,
                            model_name="shop_identifier",
                        )
                    ]

        # Category
        category = result.get("receipt_category")
        if category:
            confidence = result.get("category_confidence", 0.5)
            source = result.get("category_source", "category")
            suggestions["category"] = [
                AISuggestion(
                    question=str(category),
                    probability=float(confidence),
                    model_name=str(source),
                )
            ]

        # Totals
        subtotal = result.get("subtotal")
        if subtotal is not None:
            suggestions["subtotal"] = [
                AISuggestion(
                    question=str(subtotal),
                    probability=0.80,
                    model_name="totals",
                )
            ]

        total_tax = result.get("total_tax")
        if total_tax is not None:
            suggestions["total_tax"] = [
                AISuggestion(
                    question=str(total_tax),
                    probability=0.80,
                    model_name="totals",
                )
            ]

        # Line items
        for list_key in ("net_bought_items", "net_returned_items"):
            items = result.get(list_key)
            if not isinstance(items, list):
                continue
            prefix = (
                "bought" if list_key == "net_bought_items" else "returned"
            )
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                for field in (
                    "description",
                    "quantity",
                    "unit_price",
                ):
                    val = item.get(field)
                    if val is not None:
                        key = f"{prefix}_{i}_{field}"
                        suggestions[key] = [
                            AISuggestion(
                                question=str(val),
                                probability=0.75,
                                model_name="line_items",
                            )
                        ]

        # Cross-check metadata (informational)
        checks = result.get("_checks")
        if isinstance(checks, dict):
            suggestions["_checks"] = [
                AISuggestion(
                    question=str(checks),
                    probability=0.0,
                    model_name="_checks",
                )
            ]

        return suggestions
