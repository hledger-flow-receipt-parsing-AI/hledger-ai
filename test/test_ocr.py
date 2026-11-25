"""OCR results: visual rows, the on-disk cache and the real engine."""

from pathlib import Path
from typing import List

import pytest
from PIL import Image, ImageDraw, ImageFont

from hledger_ai.ocr.cache import OcrCache, ocr_image
from hledger_ai.ocr.types import OcrLine, OcrResult


def _line(
    text: str, x: float, y: float, w: float = 80, h: float = 30
) -> OcrLine:
    return OcrLine(
        text=text,
        score=0.99,
        box=((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
    )


def _result(lines: List[OcrLine]) -> OcrResult:
    return OcrResult(
        engine="fake",
        image_sha256="0" * 64,
        width=600,
        height=900,
        elapsed_s=0.1,
        lines=tuple(lines),
    )


def test_boxes_on_one_printed_row_are_joined_left_to_right() -> None:
    result = _result(
        [
            _line("11,54", x=224, y=392),
            _line("PIN", x=36, y=436),
            _line("TOTAAL", x=37, y=393),
            _line("11,54", x=224, y=436),
        ]
    )

    assert result.row_texts() == ["TOTAAL 11,54", "PIN 11,54"]


def test_slightly_skewed_row_still_counts_as_one_row() -> None:
    # A photographed receipt is rarely level: the right end sits lower.
    result = _result(
        [_line("TOTAAL", x=40, y=100), _line("11,54", x=400, y=111)]
    )

    assert result.row_texts() == ["TOTAAL 11,54"]


def test_cache_returns_what_was_stored(tmp_path: Path) -> None:
    cache = OcrCache(cache_dir=str(tmp_path))
    stored = _result([_line("TOTAAL", x=37, y=393)])

    cache.put(result=stored)

    assert cache.get(engine="fake", image_sha256="0" * 64) == stored
    assert cache.get(engine="other", image_sha256="0" * 64) is None


class _CountingEngine:
    engine_id = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult:
        self.calls += 1
        return OcrResult(
            engine=self.engine_id,
            image_sha256=image_sha256,
            width=10,
            height=10,
            elapsed_s=0.0,
            lines=(),
        )


def test_same_image_content_is_read_by_the_engine_only_once(
    tmp_path: Path,
) -> None:
    first, copy = tmp_path / "a.png", tmp_path / "copy_of_a.png"
    Image.new("RGB", (10, 10), "white").save(first)
    copy.write_bytes(first.read_bytes())
    engine = _CountingEngine()
    cache = OcrCache(cache_dir=str(tmp_path / "cache"))

    ocr_image(image_path=str(first), engine=engine, cache=cache)
    ocr_image(image_path=str(copy), engine=engine, cache=cache)

    assert engine.calls == 1


def test_changed_image_content_is_read_again(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    engine = _CountingEngine()
    cache = OcrCache(cache_dir=str(tmp_path / "cache"))

    Image.new("RGB", (10, 10), "white").save(path)
    ocr_image(image_path=str(path), engine=engine, cache=cache)
    Image.new("RGB", (10, 10), "black").save(path)
    ocr_image(image_path=str(path), engine=engine, cache=cache)

    assert engine.calls == 2


@pytest.fixture
def printed_receipt(tmp_path: Path) -> str:
    """An invented receipt drawn with a monospace font."""
    image = Image.new("RGB", (600, 500), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26
    )
    rows = ["VOORBEELD MARKT", "1234 AB Voorbeeldstad", "TOTAAL      11,54"]
    for index, row in enumerate(rows):
        draw.text((40, 40 + 60 * index), row, fill="black", font=font)
    path = tmp_path / "receipt.png"
    image.save(path)
    return str(path)


def test_rapidocr_reads_a_printed_receipt(
    printed_receipt: str, tmp_path: Path
) -> None:
    pytest.importorskip("rapidocr")
    from hledger_ai.ocr.rapidocr_engine import RapidOcrEngine

    result = ocr_image(
        image_path=printed_receipt,
        engine=RapidOcrEngine(),
        cache=OcrCache(cache_dir=str(tmp_path / "cache")),
    )

    assert result.row_texts() == [
        "VOORBEELD MARKT",
        "1234 AB Voorbeeldstad",
        "TOTAAL 11,54",
    ]
    assert (result.width, result.height) == (600, 500)


def _tilted(text: str, x: float, y: float, degrees: float) -> OcrLine:
    """A 200x30 box whose corners are rotated around the image origin."""
    import math

    angle = math.radians(degrees)

    def turn(px: float, py: float) -> tuple:
        return (
            px * math.cos(angle) - py * math.sin(angle),
            px * math.sin(angle) + py * math.cos(angle),
        )

    corners = [(x, y), (x + 200, y), (x + 200, y + 30), (x, y + 30)]
    return OcrLine(text=text, score=0.99, box=tuple(turn(*c) for c in corners))


def test_rows_survive_a_receipt_photographed_at_an_angle() -> None:
    # At 4 degrees the right-hand column sits ~60 px lower than the left,
    # more than a full text row.
    result = _result(
        [
            _tilted("11,54", x=900, y=100, degrees=4),
            _tilted("TOTAAL", x=40, y=100, degrees=4),
            _tilted("0,95", x=900, y=145, degrees=4),
            _tilted("BTW", x=40, y=145, degrees=4),
        ]
    )

    assert result.row_texts() == ["TOTAAL 11,54", "BTW 0,95"]
