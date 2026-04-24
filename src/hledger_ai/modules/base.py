"""Protocols for extraction modules and cross-checks."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ExtractionModule(Protocol):
    """Extracts a subset of fields from a receipt image.

    Each module is responsible for one logical group of fields
    (e.g. shop identity, line items, totals, date/time, category).
    Modules can use any backend: VLM, OCR, lookup table, regex, etc.
    """

    name: str

    def extract(
        self,
        *,
        image_path: str,
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return extracted fields as a dict.

        ``existing`` contains merged outputs from all modules that ran
        before this one.  Modules must not modify ``existing``.
        """
        ...

    def get_schema(self) -> Dict[str, Any]:
        """JSON Schema describing this module's output fields."""
        ...


@runtime_checkable
class CrossCheck(Protocol):
    """Validates / enriches the merged output of all extraction modules.

    Cross-checks are pure functions over the merged dict.  They may
    adjust confidence scores, flag warnings, or fill in fields using
    information from multiple modules combined.  They must not call
    AI models — use lookup tables, previous labels, and simple logic.
    """

    name: str

    def apply(
        self, merged: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return a (possibly modified) copy of *merged*."""
        ...
