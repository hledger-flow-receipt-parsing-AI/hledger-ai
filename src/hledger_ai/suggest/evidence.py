"""What is known about a receipt before anybody answers a question."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Hashable, List, Optional, Tuple

from typeguard import typechecked

from hledger_ai.candidates.amounts import AmountCandidate, amount_candidates
from hledger_ai.candidates.dates import (
    DateCandidate,
    TimeCandidate,
    date_candidates,
    time_candidates,
)
from hledger_ai.ocr.types import OcrResult


@typechecked
@dataclass(frozen=True)
class ReceiptEvidence:
    ocr: Tuple[OcrResult, ...]  # one result per photo of the receipt
    captured_at: Optional[datetime] = None  # when it was photographed

    def memo(self, *, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Work derived from this evidence, done once and remembered."""
        cache = self.__dict__.setdefault("_memo", {})
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def text(self) -> str:
        """All printed rows of all photos, one row per line."""
        return self.memo(
            key="text",
            compute=lambda: "\n".join(
                row for result in self.ocr for row in result.row_texts()
            ),
        )

    def amounts(self) -> List[AmountCandidate]:
        return self.memo(
            key="amounts", compute=lambda: amount_candidates(results=self.ocr)
        )

    def dates(self) -> List[DateCandidate]:
        return self.memo(
            key="dates",
            compute=lambda: date_candidates(
                results=self.ocr, captured_at=self.captured_at
            ),
        )

    def times(self) -> List[TimeCandidate]:
        return self.memo(
            key="times", compute=lambda: time_candidates(results=self.ocr)
        )
