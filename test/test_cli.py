"""The hledger-ai command line."""

from pathlib import Path
from typing import List

import pytest

from hledger_ai.cli import build_parser, format_progress, run
from hledger_ai.hardware import Device
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.training.state import ModelState, StrategyState


def _paths(tmp_path: Path):
    root = tmp_path / "finance"
    for name in ("in", "proc", "labels"):
        (root / name).mkdir(parents=True)
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


def test_train_without_names_means_everything_out_of_date() -> None:
    args = build_parser().parse_args(["--config", "c.yaml", "--train"])

    assert args.train == []


def test_train_accepts_models_and_strategies_by_name() -> None:
    args = build_parser().parse_args(
        ["--config", "c.yaml", "--train", "priors", "ST-1"]
    )

    assert args.train == ["priors", "ST-1"]


def test_no_train_flag_means_no_training() -> None:
    args = build_parser().parse_args(["--config", "c.yaml", "--models"])

    assert args.train is None


def test_unknown_training_name_fails_with_the_valid_names(
    tmp_path: Path,
) -> None:
    finance, ai = _paths(tmp_path)
    lines: List[str] = []
    args = build_parser().parse_args(
        ["--config", "c.yaml", "--train", "no-such-model"]
    )

    code = run(args=args, finance=finance, ai=ai, out=lines.append)

    assert code == 2
    assert "no-such-model" in "\n".join(lines)
    assert "rapidocr-linear" in "\n".join(lines)


def test_training_priors_on_an_empty_finance_root_reports_and_succeeds(
    tmp_path: Path,
) -> None:
    finance, ai = _paths(tmp_path)
    lines: List[str] = []
    args = build_parser().parse_args(
        ["--config", "c.yaml", "--train", "priors"]
    )

    code = run(args=args, finance=finance, ai=ai, out=lines.append)

    assert code == 0
    assert any("priors" in line and "trained" in line for line in lines)


def test_progress_line_shows_ocr_fits_and_remaining_time() -> None:
    state = ModelState(
        model="rapidocr-linear",
        dataset_hash="abc",
        private=True,
        model_version="2",
        hyperparameters={},
        ocr_done=50,
        ocr_total=100,
        ocr_read=50,
        ocr_seconds=100.0,
        fits_done=1,
        fits_total=5,
        fit_seconds=10.0,
    )

    line = format_progress(state=state)

    assert line == "rapidocr-linear: photos 50/100, fits 1/5, 44%, 2m left"


def test_status_once_prints_the_table_of_the_newest_configuration(
    tmp_path: Path,
) -> None:
    import io

    finance, ai = _paths(tmp_path)
    parser = build_parser()
    run(
        args=parser.parse_args(["--config", "c.yaml", "--train", "priors"]),
        finance=finance,
        ai=ai,
        out=lambda line: None,
    )
    stream = io.StringIO()

    code = run(
        args=parser.parse_args(["--config", "c.yaml", "--status", "--once"]),
        finance=finance,
        ai=ai,
        out=lambda line: None,
        stream=stream,
    )

    text = stream.getvalue()
    assert code == 0
    assert "dataset 1/1" in text
    assert any(line.startswith("priors") for line in text.splitlines())
    assert "ST-1" in text


def test_status_before_any_training_says_so(tmp_path: Path) -> None:
    import io

    finance, ai = _paths(tmp_path)
    stream = io.StringIO()

    run(
        args=build_parser().parse_args(
            ["--config", "c.yaml", "--status", "--once"]
        ),
        finance=finance,
        ai=ai,
        out=lambda line: None,
        stream=stream,
    )

    assert "Nothing has been trained yet" in stream.getvalue()


def test_models_lists_every_model_with_needed_and_available(
    tmp_path: Path,
) -> None:
    finance, ai = _paths(tmp_path)
    lines: List[str] = []

    code = run(
        args=build_parser().parse_args(["--config", "c.yaml", "--models"]),
        finance=finance,
        ai=ai,
        out=lines.append,
        device=Device(
            ram_gb=8.0,
            vram_gb=0.0,
            has_gpu=False,
            disk_free_gb=50.0,
            runtimes={"python"},
        ),
    )

    text = "\n".join(lines)
    assert code == 0
    for name in ("priors", "rapidocr-linear", "qvac-ocr-linear", "qwen-lora"):
        assert name in text
    assert "needed/available" in text
    # Only plain Python is installed on that device.
    assert "Used by default on this device: priors" in text
    assert "!onnxruntime" in text


@pytest.fixture
def plugged_in(monkeypatch) -> None:
    """The tests must not depend on the charger or change the priority of
    the test process."""
    monkeypatch.setattr("hledger_ai.cli.lower_priority", lambda: None)
    monkeypatch.setattr("hledger_ai.cli.power_pause_reason", lambda: None)
    monkeypatch.setattr(
        "hledger_ai.background.power_pause_reason", lambda: None
    )


def test_infer_reads_the_unlabelled_receipts_and_reports_how_many_are_ready(
    tmp_path: Path, plugged_in: None
) -> None:
    from PIL import Image
    from test_background import CountingOcr

    finance, ai = _paths(tmp_path)
    for name in ("a", "b"):
        Image.new("RGB", (4, 4), "white").save(
            Path(finance.input_dir) / f"{name}.jpg"
        )
        Image.new("RGB", (4, 4), (len(name), ord(name), 0)).save(
            Path(finance.processed_dir) / f"{name}_cropped.jpg"
        )
    ocr = CountingOcr()
    lines: List[str] = []

    code = run(
        args=build_parser().parse_args(["--config", "c.yaml", "--infer"]),
        finance=finance,
        ai=ai,
        out=lines.append,
        ocr_engines={"rapidocr-default": ocr},
    )

    assert code == 0
    assert len(ocr.read_paths) == 2
    assert "2 of 2 receipts without a label are ready" in lines[-1]


def test_install_service_writes_the_units_and_says_how_to_enable_them(
    tmp_path: Path,
) -> None:
    finance, ai = _paths(tmp_path)
    lines: List[str] = []
    units = tmp_path / "systemd"

    code = run(
        args=build_parser().parse_args(
            [
                "--config",
                "/data/finance/config.yaml",
                "--install-service",
                "--service-dir",
                str(units),
            ]
        ),
        finance=finance,
        ai=ai,
        out=lines.append,
    )

    assert code == 0
    service = (units / "hledger-ai-background.service").read_text()
    assert "--config /data/finance/config.yaml --train --infer" in service
    assert (units / "hledger-ai-background.timer").exists()
    assert any(
        "systemctl --user enable --now hledger-ai-background.timer" in line
        for line in lines
    )


def test_infer_on_battery_waits_unless_told_to_continue(
    tmp_path: Path, monkeypatch
) -> None:
    from PIL import Image
    from test_background import CountingOcr

    monkeypatch.setattr("hledger_ai.cli.lower_priority", lambda: None)
    monkeypatch.setattr(
        "hledger_ai.cli.power_pause_reason", lambda: "on battery"
    )
    monkeypatch.setattr(
        "hledger_ai.background.power_pause_reason", lambda: "on battery"
    )
    finance, ai = _paths(tmp_path)
    Image.new("RGB", (4, 4), "white").save(Path(finance.input_dir) / "a.jpg")
    Image.new("RGB", (4, 4), "black").save(
        Path(finance.processed_dir) / "a_cropped.jpg"
    )
    ocr = CountingOcr()
    lines: List[str] = []

    run(
        args=build_parser().parse_args(
            ["--config", "c.yaml", "--infer", "--on-battery"]
        ),
        finance=finance,
        ai=ai,
        out=lines.append,
        ocr_engines={"rapidocr-default": ocr},
    )

    assert len(ocr.read_paths) == 1
    assert not any("waits for the charger" in line for line in lines)
