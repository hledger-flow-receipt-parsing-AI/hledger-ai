"""Extract date and time from a receipt image."""

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
class DateTimeModule:
    """Extracts the receipt date via VLM."""

    name = "date_time"

    FIELDS = [
        "the_date",
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
                "Focus on the date and time printed on this receipt. "
                "Return an ISO 8601 formatted date-time string "
                "(YYYY-MM-DDTHH:MM:SS). Look for common date formats "
                "like DD-MM-YYYY, DD/MM/YYYY, or YYYY-MM-DD."
            ),
        )
        return self._client.generate_structured(
            model=self._model,
            prompt=prompt,
            image_path=image_path,
            schema=self._schema,
        )
