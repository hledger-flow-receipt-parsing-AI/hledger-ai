"""OCR through the qvac runtime (HW-8)."""

from __future__ import annotations

import time
from typing import Optional

from PIL import Image
from typeguard import typechecked

from hledger_ai.ocr.types import OcrLine, OcrResult
from hledger_ai.runtimes.qvac_client import QvacClient


@typechecked
class QvacOcrEngine:
    def __init__(
        self,
        *,
        engine_id: str = "qvac-ocr-latin",
        client: Optional[QvacClient] = None,
    ) -> None:
        self.engine_id = engine_id
        self._client = client or QvacClient()

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult:
        with Image.open(image_path) as image:
            width, height = image.size
        started = time.monotonic()
        blocks = self._client.ocr(image_path=image_path)
        elapsed = time.monotonic() - started
        lines = []
        for block in blocks:
            left, top, right, bottom = (float(v) for v in block["bbox"])
            lines.append(
                OcrLine(
                    text=str(block["text"]),
                    score=float(block.get("confidence", 0.0)),
                    box=(
                        (left, top),
                        (right, top),
                        (right, bottom),
                        (left, bottom),
                    ),
                )
            )
        return OcrResult(
            engine=self.engine_id,
            image_sha256=image_sha256,
            width=width,
            height=height,
            elapsed_s=elapsed,
            lines=tuple(lines),
        )
