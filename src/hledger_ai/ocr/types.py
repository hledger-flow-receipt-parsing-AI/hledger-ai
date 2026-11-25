"""OCR output: text boxes of one image, grouped into printed rows."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable

from typeguard import typechecked

Point = Tuple[float, float]


@typechecked
@dataclass(frozen=True)
class OcrLine:
    """One recognised text box; *box* is its four corners, clockwise from
    the top left."""

    text: str
    score: float
    box: Tuple[Point, Point, Point, Point]

    def centre(self) -> Point:
        return (
            sum(p[0] for p in self.box) / 4.0,
            sum(p[1] for p in self.box) / 4.0,
        )

    def height(self) -> float:
        left = math.dist(self.box[0], self.box[3])
        right = math.dist(self.box[1], self.box[2])
        return (left + right) / 2.0

    def width(self) -> float:
        return math.dist(self.box[0], self.box[1])

    def angle(self) -> float:
        """Slope of the top edge (radians)."""
        dx = self.box[1][0] - self.box[0][0]
        dy = self.box[1][1] - self.box[0][1]
        return math.atan2(dy, dx)


@typechecked
@dataclass(frozen=True)
class OcrResult:
    engine: str
    image_sha256: str
    width: int
    height: int
    elapsed_s: float
    lines: Tuple[OcrLine, ...]

    def rows(self) -> List[List[OcrLine]]:
        """Boxes grouped into printed rows, top to bottom, left to right.

        The receipt is rarely level in a photo, so box centres are first
        levelled with the typical slope of the boxes. Worked out once.
        """
        if "_rows" not in self.__dict__:
            self.__dict__["_rows"] = self._group_rows()
        return self.__dict__["_rows"]

    def _group_rows(self) -> List[List[OcrLine]]:
        if not self.lines:
            return []
        wide = [l.angle() for l in self.lines if l.width() > 2 * l.height()]
        slope = math.tan(statistics.median(wide)) if wide else 0.0
        levelled = sorted(
            ((l.centre()[1] - slope * l.centre()[0], l) for l in self.lines),
            key=lambda pair: pair[0],
        )
        rows: List[List[Tuple[float, OcrLine]]] = []
        for y, line in levelled:
            if rows:
                row_y = statistics.fmean(v for v, _ in rows[-1])
                row_h = min(l.height() for _, l in rows[-1])
                if abs(y - row_y) <= 0.5 * min(row_h, line.height()):
                    rows[-1].append((y, line))
                    continue
            rows.append([(y, line)])
        return [
            sorted((l for _, l in row), key=lambda l: l.centre()[0])
            for row in rows
        ]

    def row_texts(self) -> List[str]:
        if "_row_texts" not in self.__dict__:
            self.__dict__["_row_texts"] = [
                " ".join(l.text for l in row) for row in self.rows()
            ]
        return self.__dict__["_row_texts"]

    def to_json(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "elapsed_s": self.elapsed_s,
            "lines": [
                {
                    "text": l.text,
                    "score": l.score,
                    "box": [list(p) for p in l.box],
                }
                for l in self.lines
            ],
        }

    @staticmethod
    def from_json(*, data: Dict[str, Any]) -> "OcrResult":
        return OcrResult(
            engine=data["engine"],
            image_sha256=data["image_sha256"],
            width=int(data["width"]),
            height=int(data["height"]),
            elapsed_s=float(data["elapsed_s"]),
            lines=tuple(
                OcrLine(
                    text=l["text"],
                    score=float(l["score"]),
                    box=tuple(  # type: ignore[arg-type]
                        (float(x), float(y)) for x, y in l["box"]
                    ),
                )
                for l in data["lines"]
            ),
        )


@runtime_checkable
class OcrEngine(Protocol):
    """Reads the text of one image. Each engine declares its runtime in the
    model registry (HW-8)."""

    engine_id: str

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult: ...
