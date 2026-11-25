"""Private on-disk OCR cache, keyed by engine and image content.

OCR text is private data: the cache directory is created owner-only and
nothing in it leaves the device (PR-1). A cached image is never read by the
engine again, which is what lets background inference resume (BG-1).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from typeguard import typechecked

from hledger_ai.ocr.types import OcrEngine, OcrResult


@typechecked
def file_sha256(*, path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@typechecked
class OcrCache:
    def __init__(self, *, cache_dir: str) -> None:
        self._dir = os.path.expanduser(cache_dir)

    def _path(self, *, engine: str, image_sha256: str) -> str:
        return os.path.join(self._dir, engine, f"{image_sha256}.json")

    def get(self, *, engine: str, image_sha256: str) -> Optional[OcrResult]:
        path = self._path(engine=engine, image_sha256=image_sha256)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return OcrResult.from_json(data=json.load(handle))

    def put(self, *, result: OcrResult) -> None:
        path = self._path(
            engine=result.engine, image_sha256=result.image_sha256
        )
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(result.to_json(), handle)
        os.replace(temporary, path)


@typechecked
def ocr_image(
    *, image_path: str, engine: OcrEngine, cache: OcrCache
) -> OcrResult:
    """OCR of *image_path*, from the cache when its content was read before."""
    sha = file_sha256(path=image_path)
    cached = cache.get(engine=engine.engine_id, image_sha256=sha)
    if cached is not None:
        return cached
    result = engine.read(image_path=image_path, image_sha256=sha)
    cache.put(result=result)
    return result
