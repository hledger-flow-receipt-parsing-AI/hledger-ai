"""Extract line items (bought/returned) from a receipt image."""

from __future__ import annotations

from typing import Any, Dict

from typeguard import typechecked

from hledger_ai.ollama_client import OllamaClient
from hledger_ai.schemas.schema_generator import (
    build_extraction_prompt,
    generate_extraction_schema,
)
from hledger_core.TransactionObjects.Receipt import Receipt


@typechecked
class LineItemsModule:
    """Extracts bought/returned item details via VLM."""

    name = "line_items"

    FIELDS = [
        "net_bought_items.quantity",
        "net_bought_items.description",
        "net_bought_items.unit_price",
        "net_bought_items.tax_per_unit",
        "net_bought_items.group_discount",
        "net_bought_items.category",
        "net_returned_items.quantity",
        "net_returned_items.description",
        "net_returned_items.unit_price",
        "net_returned_items.tax_per_unit",
        "net_returned_items.group_discount",
        "net_returned_items.category",
    ]

    def __init__(
        self,
        *,
        client: OllamaClient,
        model: str = "qwen3-vl:2b",
    ) -> None:
        self._client = client
        self._model = model
        self._schema = generate_extraction_schema(
            source_class=Receipt, fields=self.FIELDS
        )

    def get_schema(self) -> Dict[str, Any]:
        return self._schema

    def extract(
        self,
        *,
        image_path: str,
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = build_extraction_prompt(
            schema=self._schema,
            context=(
                "Focus on the individual line items on this receipt. "
                "Extract each item with its quantity, description, "
                "unit price, and any applicable tax or discount. "
                "Separate bought items from returned/refunded items."
            ),
        )
        return self._client.generate_structured(
            model=self._model,
            prompt=prompt,
            image_path=image_path,
            schema=self._schema,
        )
