"""The models that exist, what they need (HW-1), which class they are in
(HW-2) and which runtime runs them (HW-8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from typeguard import typechecked

from hledger_ai.ocr.types import OcrEngine


@typechecked
@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: int  # TR-1: 1 priors, 2 small classifiers, 3 fine-tuned
    runtime: str  # HW-8: python, onnxruntime, qvac, ollama or torch
    weight_class: str  # HW-2: "light" runs on any old laptop, else "heavy"
    ram_gb: float
    vram_gb: float
    disk_gb: float
    needs_gpu: bool
    uses_image: bool
    ocr_engine: Optional[str] = None  # which OCR engine reads the photos
    trained_here: bool = True  # False: listed, but trained by another tool
    note: str = ""


@typechecked
@dataclass(frozen=True)
class OcrEngineSpec:
    engine_id: str
    runtime: str
    params: Dict[str, Any] = field(default_factory=dict)


OCR_ENGINES: Tuple[OcrEngineSpec, ...] = (
    OcrEngineSpec(engine_id="rapidocr-default", runtime="onnxruntime"),
    OcrEngineSpec(
        engine_id="rapidocr-side4000",
        runtime="onnxruntime",
        params={"Global.max_side_len": 4000},
    ),
    OcrEngineSpec(engine_id="qvac-ocr-latin", runtime="qvac"),
)

MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="priors",
        tier=1,
        runtime="python",
        weight_class="light",
        ram_gb=0.2,
        vram_gb=0.0,
        disk_gb=0.01,
        needs_gpu=False,
        uses_image=False,
        note="frequency tables over confirmed fields, no image needed",
    ),
    ModelSpec(
        name="rapidocr-linear",
        tier=2,
        runtime="onnxruntime",
        weight_class="light",
        ram_gb=1.5,
        vram_gb=0.0,
        disk_gb=0.2,
        needs_gpu=False,
        uses_image=True,
        ocr_engine="rapidocr-default",
        note="PP-OCR small on the CPU, linear models on its text",
    ),
    ModelSpec(
        name="rapidocr-hires-linear",
        tier=2,
        runtime="onnxruntime",
        weight_class="heavy",
        ram_gb=4.0,
        vram_gb=0.0,
        disk_gb=0.2,
        needs_gpu=False,
        uses_image=True,
        ocr_engine="rapidocr-side4000",
        note="same, reading photos at twice the resolution",
    ),
    ModelSpec(
        name="qvac-ocr-linear",
        tier=2,
        runtime="qvac",
        weight_class="heavy",
        ram_gb=4.0,
        vram_gb=0.0,
        disk_gb=3.5,
        needs_gpu=False,
        uses_image=True,
        ocr_engine="qvac-ocr-latin",
        note="OCR through the qvac SDK, linear models on its text",
    ),
    ModelSpec(
        name="ollama-vlm",
        tier=3,
        runtime="ollama",
        weight_class="heavy",
        ram_gb=8.0,
        vram_gb=4.0,
        disk_gb=4.0,
        needs_gpu=False,
        uses_image=True,
        trained_here=False,
        note="pretrained vision model, not trained on your labels",
    ),
    ModelSpec(
        name="setfit",
        tier=2,
        runtime="torch",
        weight_class="heavy",
        ram_gb=4.0,
        vram_gb=0.0,
        disk_gb=3.0,
        needs_gpu=False,
        uses_image=False,
        trained_here=False,
        note="bank transaction text to category (--train-models)",
    ),
    ModelSpec(
        name="qwen-lora",
        tier=3,
        runtime="torch",
        weight_class="heavy",
        ram_gb=16.0,
        vram_gb=6.0,
        disk_gb=6.0,
        needs_gpu=True,
        uses_image=False,
        trained_here=False,
        note="fine-tuned language model (--train-models)",
    ),
)


@typechecked
def get_model(*, name: str) -> ModelSpec:
    for spec in MODELS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model: {name}")


@typechecked
def build_ocr_engine(*, engine_id: str) -> OcrEngine:
    """The OCR engine of that id, loaded lazily on its first image."""
    for spec in OCR_ENGINES:
        if spec.engine_id != engine_id:
            continue
        if spec.runtime == "onnxruntime":
            from hledger_ai.ocr.rapidocr_engine import RapidOcrEngine

            return RapidOcrEngine(engine_id=engine_id, params=spec.params)
        if spec.runtime == "qvac":
            from hledger_ai.ocr.qvac_engine import QvacOcrEngine

            return QvacOcrEngine(engine_id=engine_id)
    raise KeyError(f"Unknown OCR engine: {engine_id}")
