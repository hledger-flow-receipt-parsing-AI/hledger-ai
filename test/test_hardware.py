"""Hardware requirements, model listing and default selection (HW-*)."""

from hledger_ai.hardware import (
    Device,
    check_requirements,
    render_model_table,
    runs_flawlessly,
    select_models,
)
from hledger_ai.models.catalogue import ModelSpec


def _spec(**changes) -> ModelSpec:
    values = dict(
        name="example",
        tier=2,
        runtime="onnxruntime",
        weight_class="light",
        ram_gb=4.0,
        vram_gb=0.0,
        disk_gb=1.0,
        needs_gpu=False,
        uses_image=True,
    )
    values.update(changes)
    return ModelSpec(**values)


def _device(**changes) -> Device:
    values = dict(
        ram_gb=16.0,
        vram_gb=4.0,
        has_gpu=True,
        disk_free_gb=100.0,
        runtimes={"python", "onnxruntime"},
    )
    values.update(changes)
    return Device(**values)


def _unmet(spec: ModelSpec, device: Device) -> list:
    return [
        r.name
        for r in check_requirements(spec=spec, device=device)
        if not r.met
    ]


def test_model_that_fits_has_no_unmet_requirement() -> None:
    assert _unmet(_spec(), _device()) == []


def test_ram_needs_a_safety_margin_on_top_of_the_requirement() -> None:
    # 4 GB needed: 4.5 GB installed is too tight, 5 GB is enough.
    assert _unmet(_spec(ram_gb=4.0), _device(ram_gb=4.5)) == ["ram"]
    assert _unmet(_spec(ram_gb=4.0), _device(ram_gb=5.0)) == []


def test_missing_gpu_memory_disk_and_runtime_are_each_reported() -> None:
    spec = _spec(vram_gb=6.0, needs_gpu=True, disk_gb=50.0, runtime="ollama")
    device = _device(vram_gb=4.0, disk_free_gb=20.0)

    assert _unmet(spec, device) == ["vram", "disk", "runtime"]


def test_gpu_requirement_fails_on_a_machine_without_gpu() -> None:
    spec = _spec(needs_gpu=True, vram_gb=2.0)

    assert _unmet(spec, _device(has_gpu=False, vram_gb=0.0)) == ["vram", "gpu"]


def test_flawless_needs_met_requirements_and_a_fast_enough_benchmark() -> None:
    spec, device = _spec(), _device()

    assert runs_flawlessly(spec=spec, device=device, benchmark_seconds=2.0)
    assert not runs_flawlessly(spec=spec, device=device, benchmark_seconds=40.0)
    assert not runs_flawlessly(
        spec=_spec(ram_gb=64.0), device=device, benchmark_seconds=2.0
    )
    # Not benchmarked (e.g. the tier-1 priors): requirements decide.
    assert runs_flawlessly(spec=spec, device=device, benchmark_seconds=None)


def test_listing_shows_need_next_to_what_the_device_has_with_unmet_in_red() -> (
    None
):
    table = render_model_table(
        specs=[
            _spec(name="fits"),
            _spec(name="too-big", vram_gb=6.0, needs_gpu=True),
        ],
        device=_device(vram_gb=4.0),
        colour=True,
    )
    lines = table.splitlines()
    fits = next(line for line in lines if line.startswith("fits"))
    too_big = next(line for line in lines if line.startswith("too-big"))

    assert "\x1b[31m" not in fits
    assert "\x1b[31m6.0/4.0\x1b[0m" in too_big  # needed/available VRAM
    assert "4.0/16.0" in too_big  # RAM is fine, so not red
    assert "1.0/100.0" in too_big  # needed and available disk space (HW-6)


def test_listing_without_colour_marks_unmet_requirements_in_text() -> None:
    table = render_model_table(
        specs=[_spec(name="too-big", vram_gb=6.0)],
        device=_device(vram_gb=4.0),
        colour=False,
    )

    assert "\x1b[" not in table
    assert "!6.0/4.0" in table
    assert "20% safety margin" in table


def test_default_selection_keeps_only_models_that_run_flawlessly() -> None:
    specs = [
        _spec(name="small"),
        _spec(name="needs-gpu", needs_gpu=True, vram_gb=8.0),
        _spec(name="slow"),
    ]

    chosen = select_models(
        specs=specs,
        device=_device(),
        benchmarks={"small": 1.0, "slow": 60.0},
        accuracy={},
    )

    assert chosen == ["small"]


def test_when_disk_cannot_hold_all_models_the_most_accurate_are_kept() -> None:
    specs = [
        _spec(name="a", disk_gb=6.0),
        _spec(name="b", disk_gb=6.0),
        _spec(name="c", disk_gb=6.0),
    ]

    chosen = select_models(
        specs=specs,
        device=_device(disk_free_gb=15.0),
        benchmarks={},
        accuracy={"a": 0.6, "b": 0.9, "c": 0.8},
    )

    # 15 GB free minus the safety margin holds two 6 GB models.
    assert chosen == ["b", "c"]


def test_without_known_accuracy_cheaper_tiers_win_the_disk_space() -> None:
    specs = [
        _spec(name="tier3", tier=3, disk_gb=9.0),
        _spec(name="tier1", tier=1, disk_gb=9.0),
    ]

    chosen = select_models(
        specs=specs,
        device=_device(disk_free_gb=12.0),
        benchmarks={},
        accuracy={},
    )

    assert chosen == ["tier1"]
