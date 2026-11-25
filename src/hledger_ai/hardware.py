"""What this device can run (requirements HW-1 to HW-7).

Each model declares what it needs; the device is probed; the listing shows
both side by side with unmet requirements in red; by default only models
that run flawlessly here are used.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from typeguard import typechecked

from hledger_ai.models.catalogue import ModelSpec, build_ocr_engine

# HW-3: requirements must be met with this much headroom.
SAFETY_MARGIN = 1.2
# HW-3: a model used while labelling may take at most this long per photo.
BENCHMARK_LIMIT_SECONDS = 10.0

_RED, _RESET = "\x1b[31m", "\x1b[0m"


@typechecked
@dataclass(frozen=True)
class Device:
    ram_gb: float
    vram_gb: float
    has_gpu: bool
    disk_free_gb: float
    runtimes: Set[str]  # runtimes installed and reachable (HW-8)


@typechecked
@dataclass(frozen=True)
class Requirement:
    name: str  # ram, vram, gpu, disk or runtime
    needed: str
    available: str
    met: bool


@typechecked
def _amount(*, name: str, needed: float, available: float) -> Requirement:
    return Requirement(
        name=name,
        needed=f"{needed:.1f}",
        available=f"{available:.1f}",
        met=needed * SAFETY_MARGIN <= available or needed == 0,
    )


@typechecked
def check_requirements(*, spec: ModelSpec, device: Device) -> List[Requirement]:
    """HW-1 against the device, in a fixed order."""
    return [
        _amount(name="ram", needed=spec.ram_gb, available=device.ram_gb),
        _amount(name="vram", needed=spec.vram_gb, available=device.vram_gb),
        Requirement(
            name="gpu",
            needed="yes" if spec.needs_gpu else "no",
            available="yes" if device.has_gpu else "no",
            met=device.has_gpu or not spec.needs_gpu,
        ),
        _amount(
            name="disk", needed=spec.disk_gb, available=device.disk_free_gb
        ),
        Requirement(
            name="runtime",
            needed=spec.runtime,
            available=spec.runtime if spec.runtime in device.runtimes else "-",
            met=spec.runtime in device.runtimes,
        ),
    ]


@typechecked
def runs_flawlessly(
    *, spec: ModelSpec, device: Device, benchmark_seconds: Optional[float]
) -> bool:
    """HW-3: every requirement met with margin and, when benchmarked, fast
    enough for use during labelling."""
    met = all(r.met for r in check_requirements(spec=spec, device=device))
    fast = (
        benchmark_seconds is None
        or benchmark_seconds <= BENCHMARK_LIMIT_SECONDS
    )
    return met and fast


@typechecked
def select_models(
    *,
    specs: Sequence[ModelSpec],
    device: Device,
    benchmarks: Dict[str, float],
    accuracy: Dict[str, float],
) -> List[str]:
    """HW-4 and HW-7: the models to use by default, best first.

    Only models that run flawlessly; when the disk cannot hold them all,
    the most accurate ones that fit are kept (cheaper tiers first while no
    accuracy is known yet).
    """
    usable = [
        spec
        for spec in specs
        if runs_flawlessly(
            spec=spec,
            device=device,
            benchmark_seconds=benchmarks.get(spec.name),
        )
    ]
    usable.sort(key=lambda s: (-accuracy.get(s.name, -1.0), s.tier, s.name))
    budget = device.disk_free_gb / SAFETY_MARGIN
    chosen: List[str] = []
    used = 0.0
    for spec in usable:
        if used + spec.disk_gb <= budget:
            chosen.append(spec.name)
            used += spec.disk_gb
    return chosen


@typechecked
def _cell(*, requirement: Requirement, colour: bool, width: int) -> str:
    if requirement.name == "gpu":
        text = requirement.needed
    elif requirement.name == "runtime":
        text = requirement.needed
    else:
        text = f"{requirement.needed}/{requirement.available}"
    if requirement.met:
        return text.ljust(width)
    if colour:
        return f"{_RED}{text}{_RESET}" + " " * max(width - len(text), 0)
    return f"!{text}".ljust(width)


@typechecked
def render_model_table(
    *, specs: Sequence[ModelSpec], device: Device, colour: bool
) -> str:
    """HW-5 and HW-6: every model with needed/available per requirement;
    unmet requirements are red (or prefixed with ``!`` without colour)."""
    header = (
        f"{'model':24}{'tier':5}{'class':7}{'runtime':13}"
        f"{'RAM GB':12}{'VRAM GB':12}{'disk GB':13}{'GPU':5}"
    )
    lines = [header, "-" * len(header)]
    for spec in specs:
        by_name = {
            r.name: r for r in check_requirements(spec=spec, device=device)
        }
        lines.append(
            f"{spec.name:24}{spec.tier:<5}{spec.weight_class:7}"
            + _cell(requirement=by_name["runtime"], colour=colour, width=13)
            + _cell(requirement=by_name["ram"], colour=colour, width=12)
            + _cell(requirement=by_name["vram"], colour=colour, width=12)
            + _cell(requirement=by_name["disk"], colour=colour, width=13)
            + _cell(requirement=by_name["gpu"], colour=colour, width=5)
        )
    lines.append("")
    lines.append(
        "needed/available; requirements include a "
        f"{round((SAFETY_MARGIN - 1) * 100)}% safety margin."
    )
    return "\n".join(lines)


@typechecked
def _vram_gb() -> float:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return max(float(line) for line in output.split()) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


@typechecked
def _ram_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


@typechecked
def probe_device(*, storage_dir: str, ollama_url: str, qvac_dir: str) -> Device:
    """Measure this machine. *storage_dir* is where models are kept."""
    runtimes = {"python"}
    if importlib.util.find_spec("rapidocr") and importlib.util.find_spec(
        "onnxruntime"
    ):
        runtimes.add("onnxruntime")
    if importlib.util.find_spec("torch"):
        runtimes.add("torch")
    if shutil.which("node") and os.path.isdir(
        os.path.join(qvac_dir, "node_modules", "@qvac", "sdk")
    ):
        runtimes.add("qvac")
    from hledger_ai.ollama_client import OllamaClient

    if OllamaClient(base_url=ollama_url).is_available():
        runtimes.add("ollama")
    existing = os.path.expanduser(storage_dir)
    while not os.path.isdir(existing):
        existing = os.path.dirname(existing) or "/"
    vram = _vram_gb()
    return Device(
        ram_gb=_ram_gb(),
        vram_gb=vram,
        has_gpu=vram > 0,
        disk_free_gb=shutil.disk_usage(existing).free / 1024.0**3,
        runtimes=runtimes,
    )


@typechecked
def benchmark_model(*, spec: ModelSpec) -> Optional[float]:
    """HW-3: seconds this device needs per photo for the model's slowest
    step (OCR). None for models that read no photos."""
    if spec.ocr_engine is None or not spec.trained_here:
        return None
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 2400), "white")
    draw = ImageDraw.Draw(image)
    for row in range(40):
        draw.text(
            (60, 40 + 58 * row), f"Voorbeeld regel {row}  12,34", fill="black"
        )
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "benchmark.png")
        image.save(path)
        engine = build_ocr_engine(engine_id=spec.ocr_engine)
        engine.read(image_path=path, image_sha256="warm-up")
        started = time.monotonic()
        engine.read(image_path=path, image_sha256="benchmark")
        return time.monotonic() - started
