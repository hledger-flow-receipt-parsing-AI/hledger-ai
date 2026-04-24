"""Categorize a receipt using lookup tables and optional LLM fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.ollama_client import OllamaClient

log = logging.getLogger(__name__)


@typechecked
class CategoryModule:
    """Assigns a category to a receipt.

    Strategy (in order):
    1. If shop name matches a known shop→category mapping from previous
       labels, use that (instant, high confidence).
    2. Otherwise, fall back to Qwen3-0.6B zero-shot classification
       using the user's category tree.
    """

    name = "category"

    def __init__(
        self,
        *,
        known_mappings: Optional[Dict[str, str]] = None,
        category_tree: Optional[List[str]] = None,
        client: Optional[OllamaClient] = None,
        model: str = "qwen3:0.6b",
    ) -> None:
        self._known: Dict[str, str] = known_mappings or {}
        self._categories: List[str] = category_tree or []
        self._client = client
        self._model = model

    def get_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "receipt_category": {"type": "string"},
                "category_confidence": {"type": "number"},
                "category_source": {"type": "string"},
            },
            "required": [
                "receipt_category",
                "category_confidence",
                "category_source",
            ],
        }
        return schema

    def extract(
        self,
        *,
        image_path: str,
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        shop_name = self._get_shop_name(existing)

        # Strategy 1: lookup table
        if shop_name:
            normalized = shop_name.strip().lower()
            for known_shop, category in self._known.items():
                if known_shop.lower() == normalized:
                    log.info(
                        "Category lookup hit: %s -> %s",
                        shop_name,
                        category,
                    )
                    return {
                        "receipt_category": category,
                        "category_confidence": 0.95,
                        "category_source": "lookup",
                    }

        # Strategy 2: LLM zero-shot fallback
        if self._client and self._categories:
            return self._classify_with_llm(
                shop_name=shop_name or "unknown",
                existing=existing,
            )

        # No strategies available
        return {
            "receipt_category": None,
            "category_confidence": 0.0,
            "category_source": "none",
        }

    def _get_shop_name(self, existing: Dict[str, Any]) -> Optional[str]:
        """Extract shop name from earlier module outputs."""
        shop_id = existing.get("shop_identifier")
        if isinstance(shop_id, dict):
            return shop_id.get("name")
        return existing.get("name")

    def _classify_with_llm(
        self,
        *,
        shop_name: str,
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use Qwen3-0.6B zero-shot to classify the receipt."""
        assert self._client is not None

        cat_list = "\n".join(f"- {c}" for c in self._categories)
        prompt = (
            "Classify this receipt into exactly one of the "
            "following categories:\n"
            f"{cat_list}\n\n"
            f"Shop name: {shop_name}\n"
        )

        # Add any extra context from other modules
        subtotal = existing.get("subtotal")
        if subtotal is not None:
            prompt += f"Total amount: {subtotal}\n"

        prompt += (
            "\nReply with only the category path, nothing else."
        )

        schema = {
            "type": "object",
            "properties": {
                "receipt_category": {"type": "string"},
            },
            "required": ["receipt_category"],
        }

        try:
            result = self._client.generate_structured(
                model=self._model,
                prompt=prompt,
                schema=schema,
            )
            category = result.get("receipt_category", "")
            confidence = (
                0.7 if category in self._categories else 0.3
            )
            return {
                "receipt_category": category,
                "category_confidence": confidence,
                "category_source": "llm_zero_shot",
            }
        except Exception:
            log.exception("LLM category classification failed")
            return {
                "receipt_category": None,
                "category_confidence": 0.0,
                "category_source": "llm_failed",
            }
