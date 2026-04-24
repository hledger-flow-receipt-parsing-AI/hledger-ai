"""Cross-check: override low-confidence category using shop history."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
class CategoryFromShop:
    """If the category module returned low confidence, but the shop name
    matches a shop with many previous labels, override the category.

    This cross-check is separate from the category module's own lookup
    so it can be independently enabled/disabled and configured with a
    different confidence threshold.
    """

    name = "category_from_shop"

    def __init__(
        self,
        *,
        shop_category_counts: Optional[Dict[str, Dict[str, int]]] = None,
        min_labels: int = 5,
        override_below_confidence: float = 0.7,
    ) -> None:
        # shop_name -> {category: count}
        self._counts: Dict[str, Dict[str, int]] = (
            shop_category_counts or {}
        )
        self._min_labels = min_labels
        self._threshold = override_below_confidence

    def apply(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(merged)

        current_conf = merged.get("category_confidence", 0.0)
        if current_conf >= self._threshold:
            return result

        shop_name = self._get_shop_name(merged)
        if not shop_name:
            return result

        normalized = shop_name.strip().lower()
        counts = self._counts.get(normalized, {})
        if not counts:
            return result

        best_category = max(counts, key=counts.get)  # type: ignore[arg-type]
        best_count = counts[best_category]

        if best_count < self._min_labels:
            return result

        total_labels = sum(counts.values())
        dominance = best_count / total_labels

        if dominance < 0.8:
            log.info(
                "Shop '%s' has no dominant category "
                "(best: %s at %.0f%%)",
                shop_name,
                best_category,
                dominance * 100,
            )
            return result

        log.info(
            "Overriding category to '%s' based on %d/%d "
            "previous labels for shop '%s'",
            best_category,
            best_count,
            total_labels,
            shop_name,
        )

        result["receipt_category"] = best_category
        result["category_confidence"] = min(
            0.9, dominance
        )
        result["category_source"] = "shop_history_override"

        result["_checks"] = result.get("_checks", {})
        result["_checks"]["category_from_shop"] = {
            "shop": shop_name,
            "overridden_to": best_category,
            "label_count": best_count,
            "total_labels": total_labels,
        }

        return result

    def _get_shop_name(
        self, merged: Dict[str, Any]
    ) -> Optional[str]:
        shop_id = merged.get("shop_identifier")
        if isinstance(shop_id, dict):
            return shop_id.get("name")
        return merged.get("name")
