"""Background reading of the receipts that still need a label (SP-1, BG-*)."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from helpers import make_label
from PIL import Image

from hledger_ai.background import (
    read_unlabelled_receipts,
    systemd_units,
)
from hledger_ai.ocr.cache import OcrCache, file_sha256
from hledger_ai.ocr.types import OcrResult
from hledger_ai.paths import AiPaths, FinancePaths


class CountingOcr:
    def __init__(self, engine_id: str = "rapidocr-default") -> None:
        self.engine_id = engine_id
        self.read_paths: List[str] = []
        self.fail_after: Optional[int] = None

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult:
        if (
            self.fail_after is not None
            and len(self.read_paths) >= self.fail_after
        ):
            raise KeyboardInterrupt
        self.read_paths.append(Path(image_path).name)
        return OcrResult(
            engine=self.engine_id,
            image_sha256=image_sha256,
            width=4,
            height=4,
            elapsed_s=0.0,
            lines=(),
        )


def _root(tmp_path: Path):
    """Four photos: a is labelled, b+c are one unlabelled receipt, d another;
    e has no cropped copy yet."""
    root = tmp_path / "finance"
    raws = {}
    for index, name in enumerate("abcde"):
        raw = root / "in" / f"{name}.jpg"
        raw.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (index, 9, 9)).save(raw)
        raws[name] = str(raw)
        if name != "e":
            crop = root / "proc" / f"{name}_cropped.jpg"
            crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), (index, 8, 8)).save(crop)
    (root / "proc" / "image_grouping.json").write_text(
        json.dumps([[raws["a"]], [raws["b"], raws["c"]]])
    )
    folder = root / "labels" / "label_a"
    folder.mkdir(parents=True)
    (folder / "receipt_image_to_obj_label.json").write_text(
        json.dumps(make_label(image_paths=[raws["a"]]))
    )
    finance = FinancePaths(
        input_dir=str(root / "in"),
        processed_dir=str(root / "proc"),
        labels_dir=str(root / "labels"),
        training_dataset_dir=str(root / "receipt_training_dataset"),
        grouping_path=str(root / "proc" / "image_grouping.json"),
        label_filename="receipt_image_to_obj_label.json",
        crop_suffix="_cropped",
        crop_ext=".jpg",
    )
    ai = AiPaths(
        models_dir=str(tmp_path / "ai" / "models"),
        feedback_dir=str(tmp_path / "ai" / "feedback"),
    )
    return finance, ai


def _run(finance, ai, ocr, **options):
    return read_unlabelled_receipts(
        finance=finance,
        ai=ai,
        ocr_engines={ocr.engine_id: ocr},
        pause_reason=options.pop("pause_reason", lambda: None),
        sleep=options.pop("sleep", lambda seconds: None),
        **options,
    )


def test_every_photo_of_every_unlabelled_receipt_is_read_once(
    tmp_path: Path,
) -> None:
    finance, ai = _root(tmp_path)
    ocr = CountingOcr()

    report = _run(finance, ai, ocr)

    assert sorted(ocr.read_paths) == [
        "b_cropped.jpg",
        "c_cropped.jpg",
        "d_cropped.jpg",
    ]
    assert (report.receipts, report.receipts_ready) == (3, 2)
    assert (report.photos_done, report.photos_total) == (3, 3)
    assert report.photos_missing == 1  # e.jpg was never cropped


def test_running_again_reads_nothing_twice(tmp_path: Path) -> None:
    finance, ai = _root(tmp_path)
    _run(finance, ai, CountingOcr())
    second = CountingOcr()

    report = _run(finance, ai, second)

    assert second.read_paths == []
    assert (report.photos_done, report.photos_total) == (3, 3)


def test_interrupted_run_continues_where_it_stopped(tmp_path: Path) -> None:
    finance, ai = _root(tmp_path)
    first = CountingOcr()
    first.fail_after = 1
    with pytest.raises(KeyboardInterrupt):
        _run(finance, ai, first)
    second = CountingOcr()

    _run(finance, ai, second)

    assert len(first.read_paths) == 1
    assert len(second.read_paths) == 2
    assert not set(first.read_paths) & set(second.read_paths)


def test_work_waits_while_the_laptop_runs_on_battery(tmp_path: Path) -> None:
    finance, ai = _root(tmp_path)
    ocr = CountingOcr()
    reasons = iter(["on battery", "on battery", None, None, None])
    naps: List[float] = []
    read_before_power: List[int] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        read_before_power.append(len(ocr.read_paths))

    _run(finance, ai, ocr, pause_reason=lambda: next(reasons), sleep=nap)

    assert len(naps) == 2
    assert read_before_power == [0, 0]  # nothing was read while paused
    assert len(ocr.read_paths) == 3


def test_progress_is_kept_in_a_state_file(tmp_path: Path) -> None:
    finance, ai = _root(tmp_path)

    _run(finance, ai, CountingOcr())

    state = json.loads(Path(ai.background_state_path).read_text())
    assert (state["photos_done"], state["photos_total"]) == (3, 3)
    assert state["receipts_ready"] == 2


def test_systemd_units_resume_the_work_after_a_reboot() -> None:
    units = systemd_units(
        python="/opt/env/bin/python", config_path="/data/finance/config.yaml"
    )

    service = units["hledger-ai-background.service"]
    timer = units["hledger-ai-background.timer"]
    assert (
        "ExecStart=/opt/env/bin/python -m hledger_ai"
        " --config /data/finance/config.yaml --train --infer" in service
    )
    assert "WorkingDirectory=/data/finance" in service
    assert "Nice=19" in service
    assert "OnBootSec=" in timer
    assert "WantedBy=timers.target" in timer


def test_reading_leaves_the_ai_folder_owner_only(tmp_path: Path) -> None:
    import os
    import stat

    finance, ai = _root(tmp_path)

    _run(finance, ai, CountingOcr())

    assert stat.S_IMODE(os.stat(ai.root).st_mode) == 0o700
