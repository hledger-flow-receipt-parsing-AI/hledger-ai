"""The user's own choice of models (HW-5)."""

from pathlib import Path
from typing import List

from test_app_training import FakeOcr, _ai, _finance_root

from hledger_ai.app import default_models, train
from hledger_ai.cli import build_parser, run
from hledger_ai.hardware import Device
from hledger_ai.selection import read_selection, write_selection
from hledger_ai.suggest.service import SuggestionService

DEVICE = Device(
    ram_gb=16.0,
    vram_gb=0.0,
    has_gpu=False,
    disk_free_gb=100.0,
    runtimes={"python"},
)


def test_without_a_choice_the_device_decides(tmp_path: Path) -> None:
    ai = _ai(tmp_path)

    assert read_selection(ai=ai) is None
    assert default_models(ai=ai, device=DEVICE) == ["priors", "csv-linear"]


def test_chosen_models_replace_the_automatic_choice(tmp_path: Path) -> None:
    ai = _ai(tmp_path)

    write_selection(ai=ai, models=["rapidocr-linear", "priors"])

    assert default_models(ai=ai, device=DEVICE) == ["rapidocr-linear", "priors"]


def test_only_chosen_models_make_suggestions(tmp_path: Path) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )

    write_selection(ai=ai, models=["priors"])

    assert list(SuggestionService.load(ai=ai).dataset_hashes()) == ["priors"]


def test_clearing_the_choice_goes_back_to_automatic(tmp_path: Path) -> None:
    ai = _ai(tmp_path)
    write_selection(ai=ai, models=["priors"])

    write_selection(ai=ai, models=[])

    assert read_selection(ai=ai) is None


def test_select_flag_stores_the_choice_and_refuses_unknown_models(
    tmp_path: Path,
) -> None:
    finance, ai = _finance_root(tmp_path, FakeOcr()), _ai(tmp_path)
    lines: List[str] = []
    parser = build_parser()

    ok = run(
        args=parser.parse_args(["--config", "c.yaml", "--select", "priors"]),
        finance=finance,
        ai=ai,
        out=lines.append,
    )
    bad = run(
        args=parser.parse_args(["--config", "c.yaml", "--select", "nope"]),
        finance=finance,
        ai=ai,
        out=lines.append,
    )

    assert (ok, bad) == (0, 2)
    assert read_selection(ai=ai) == ["priors"]
    assert "nope" in lines[-1]
