"""Extract totals and tax information from a receipt image."""

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
class TotalsModule:
    """Extracts subtotal and tax amounts via VLM."""

    name = "totals"

    FIELDS = [
        "subtotal",
        "total_tax",
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
                "Focus on the totals section of this receipt. "
                "Extract the subtotal (before tax) and the total "
                "tax amount. Look for labels like 'Subtotaal', "
                "'BTW', 'Tax', 'Total', 'Totaal'."
            ),
        )
        return self._client.generate_structured(
            model=self._model,
            prompt=prompt,
            image_path=image_path,
            schema=self._schema,
        )
