"""Cross-check: verify line item sum matches extracted total."""

from __future__ import annotations

import logging
from typing import Any, Dict

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
class TotalItemsConsistency:
    """Compares the sum of line item prices against the extracted total.

    If they match within a tolerance, flags consistency as OK.
    If they diverge significantly, flags a warning for review.
    """

    name = "total_items_consistency"

    def __init__(
        self, *, tolerance_pct: float = 1.0
    ) -> None:
        self._tolerance_pct = tolerance_pct

    def apply(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(merged)

        subtotal = merged.get("subtotal")
        items = merged.get("net_bought_items")

        if subtotal is None or not isinstance(items, list):
            return result

        item_sum = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            qty = item.get("quantity", 1)
            price = item.get("unit_price", 0)
            discount = item.get("group_discount", 0)
            if isinstance(qty, (int, float)) and isinstance(
                price, (int, float)
            ):
                line_total = qty * price
                if isinstance(discount, (int, float)):
                    line_total -= discount
                item_sum += line_total

        if item_sum == 0:
            return result

        if subtotal == 0:
            diff_pct = 100.0
        else:
            diff_pct = abs(item_sum - subtotal) / abs(subtotal) * 100

        result["_checks"] = result.get("_checks", {})
        if diff_pct <= self._tolerance_pct:
            log.info(
                "Total-items check OK: items=%.2f, "
                "subtotal=%.2f, diff=%.1f%%",
                item_sum,
                subtotal,
                diff_pct,
            )
            result["_checks"]["total_items"] = "ok"
        else:
            log.warning(
                "Total-items MISMATCH: items=%.2f, "
                "subtotal=%.2f, diff=%.1f%%",
                item_sum,
                subtotal,
                diff_pct,
            )
            result["_checks"]["total_items"] = "mismatch"
            result["_checks"]["total_items_detail"] = {
                "item_sum": round(item_sum, 2),
                "subtotal": subtotal,
                "diff_pct": round(diff_pct, 1),
            }

        return result
