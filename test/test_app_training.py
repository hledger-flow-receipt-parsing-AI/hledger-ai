"""Training from a finance root: dataset choice, names, registry (TR, DS, PR)."""

import json
from pathlib import Path
from typing import Dict, List

import pytest
from helpers import card_receipt, cash_receipt, make_label
from PIL import Image

from hledger_ai.app import (
    UnknownName,
    collect_training_data,
    resolve_names,
    train,
)
from hledger_ai.dataset.configuration import DatasetRegistry
from hledger_ai.ocr.cache import file_sha256
from hledger_ai.ocr.types import OcrResult
from hledger_ai.paths import AiPaths, FinancePaths
from hledger_ai.training.state import StateStore

TOTALS = ["11,54", "7,20", "23,05", "4,99", "15,75", "31,10", "8,40", "12,00"]


class FakeOcr:
    engine_id = "rapidocr-default"

    def __init__(self) -> None:
        self.by_sha: Dict[str, OcrResult] = {}

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult:
        return self.by_sha[image_sha256]


def _finance_root(tmp_path: Path, ocr: FakeOcr) -> FinancePaths:
    """Twelve labelled private receipts and two public partial ones."""
    root = tmp_path / "finance"
    for index, total in enumerate(TOTALS + TOTALS[:4]):
        raw = root / "in" / f"2025030{index % 9 + 1}_10000{index}.jpg"
        crop = root / "proc" / f"{raw.stem}_cropped.jpg"
        for path in (raw, crop):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), (index, 1, 0)).save(path)
        printed = f"{index + 1:02d}-03-2025 10:{index:02d}"
        evidence = card_receipt(total=total, printed_date=printed)
        sha = file_sha256(path=str(crop))
        ocr.by_sha[sha] = OcrResult(
            **{
                **evidence.ocr[0].__dict__,
                "image_sha256": sha,
                "engine": ocr.engine_id,
            }
        )
        folder = root / "labels" / f"label{index}"
        folder.mkdir(parents=True)
        (folder / "receipt_image_to_obj_label.json").write_text(
            json.dumps(
                make_label(
                    date=f"2025-03-{index + 1:02d}T10:{index:02d}:00",
                    amount=float(total.replace(",", ".")),
                    image_paths=[str(raw)],
                )
            )
        )
    for index in range(2):
        folder = root / "receipt_training_dataset" / f"public{index}"
        folder.mkdir(parents=True)
        crop = folder / f"public{index}_cropped.jpg"
        Image.new("RGB", (4, 4), (index, 2, 0)).save(crop)
        evidence = cash_receipt(total="6,35", given="20,00", back="13,65")
        sha = file_sha256(path=str(crop))
        ocr.by_sha[sha] = OcrResult(
            **{
                **evidence.ocr[0].__dict__,
                "image_sha256": sha,
                "engine": ocr.engine_id,
            }
        )
        (folder / "partial_label.json").write_text(
            json.dumps(
                {
                    "complete": False,
                    "answers": {
                        "date": f"2025-02-0{index + 1} 09:05:00",
                        "category": "food:market",
                        "amount": "20.0",
                        "change": "13.65",
                    },
                }
            )
        )
    return FinancePaths(
        input_dir=str(root / "in"),
        processed_dir=str(root / "proc"),
        labels_dir=str(root / "labels"),
        training_dataset_dir=str(root / "receipt_training_dataset"),
        grouping_path=str(root / "proc" / "image_grouping.json"),
        label_filename="receipt_image_to_obj_label.json",
        crop_suffix="_cropped",
        crop_ext=".jpg",
    )


def _ai(tmp_path: Path) -> AiPaths:
    return AiPaths(
        models_dir=str(tmp_path / "ai" / "models"),
        feedback_dir=str(tmp_path / "ai" / "feedback"),
    )


def test_private_training_data_includes_the_public_receipts(
    tmp_path: Path,
) -> None:
    finance = _finance_root(tmp_path, FakeOcr())

    data = collect_training_data(finance=finance, public_only=False)

    assert data.private is True
    assert len(data.examples) == 14


def test_public_only_training_data_holds_no_private_receipt(
    tmp_path: Path,
) -> None:
    finance = _finance_root(tmp_path, FakeOcr())

    public = collect_training_data(finance=finance, public_only=True)
    private = collect_training_data(finance=finance, public_only=False)

    assert public.private is False
    assert [e.private for e in public.examples] == [False, False]
    assert public.dataset_hash != private.dataset_hash


def test_names_are_split_into_models_and_strategies() -> None:
    models, strategies = resolve_names(
        names=["priors", "ST-1", "rapidocr-linear", "ST-12"]
    )

    assert models == ["priors", "rapidocr-linear"]
    assert strategies == {"ST-1", "ST-12"}
    assert resolve_names(names=[]) == ([], None)


def test_unknown_name_is_refused_with_the_valid_names() -> None:
    with pytest.raises(UnknownName) as error:
        resolve_names(names=["ST-1", "does-not-exist"])

    assert "does-not-exist" in str(error.value)
    assert "rapidocr-linear" in str(error.value)
    assert "ST-12" in str(error.value)


def test_training_registers_the_configuration_and_trains_named_models(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)

    states = train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )

    registered = DatasetRegistry(path=ai.registry_path).configurations()
    assert [s.model for s in states] == ["priors", "rapidocr-linear"]
    assert all(s.finished for s in states)
    assert len(registered) == 1
    assert registered[0].private is True
    assert registered[0].receipt_count == 14
    assert registered[0].dataset_hash == states[0].dataset_hash


def test_public_and_private_configurations_are_tracked_side_by_side(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)

    train(finance=finance, ai=ai, names=["priors"], folds=1, public_only=True)
    train(finance=finance, ai=ai, names=["priors"], folds=1)

    registered = DatasetRegistry(path=ai.registry_path).configurations()
    assert [c.private for c in registered] == [False, True]
    store = StateStore(models_dir=ai.models_dir)
    public_state = store.load(
        model="priors", dataset_hash=registered[0].dataset_hash
    )
    assert public_state.private is False


def test_receipts_whose_suggestions_were_corrected_weigh_double(
    tmp_path: Path,
) -> None:
    from hledger_ai.app import example_weights

    finance, ai = _finance_root(tmp_path, FakeOcr()), _ai(tmp_path)
    data = collect_training_data(finance=finance, public_only=False)
    corrected = data.examples[3]
    Path(ai.feedback_dir).mkdir(parents=True)
    Path(ai.corrections_path).write_text(
        json.dumps(
            {
                "question_id": "amount",
                "ai_suggestion": "1.00",
                "corrected_category": "4.99",
                "image_path": corrected.facts.image_paths[0],
            }
        )
        + "\n"
    )

    weights = example_weights(data=data, corrections_path=ai.corrections_path)

    assert weights == {corrected.receipt_id: 2.0}
