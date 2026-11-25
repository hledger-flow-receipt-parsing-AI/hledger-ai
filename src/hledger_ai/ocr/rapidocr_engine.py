"""RapidOCR (PP-OCR models on ONNX Runtime): runs in-process on a CPU."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from PIL import Image
from typeguard import typechecked

from hledger_ai.ocr.types import OcrLine, OcrResult


@typechecked
class RapidOcrEngine:
    """Lazy wrapper: the models load on the first image."""

    def __init__(
        self,
        *,
        engine_id: str = "rapidocr-default",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine_id = engine_id
        self._params: Dict[str, Any] = {"Global.log_level": "error"}
        self._params.update(params or {})
        self._engine: Any = None

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult:
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR(params=self._params)
        with Image.open(image_path) as image:
            width, height = image.size
        started = time.monotonic()
        output = self._engine(image_path)
        elapsed = time.monotonic() - started
        lines = []
        if output.txts is not None:
            for text, score, box in zip(
                output.txts, output.scores, output.boxes
            ):
                lines.append(
                    OcrLine(
                        text=str(text),
                        score=float(score),
                        box=tuple(  # type: ignore[arg-type]
                            (float(x), float(y)) for x, y in box
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
