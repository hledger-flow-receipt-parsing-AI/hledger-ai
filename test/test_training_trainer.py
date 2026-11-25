"""The trainer: evaluate on time-ordered folds, fit, save, resume (TR, DS, BG)."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from helpers import card_receipt, cash_receipt, make_label
from PIL import Image

from hledger_ai.dataset.configuration import dataset_hash
from hledger_ai.dataset.examples import ReceiptExample
from hledger_ai.models.catalogue import get_model
from hledger_ai.ocr.cache import OcrCache, file_sha256
from hledger_ai.ocr.types import OcrResult
from hledger_ai.receipt_fields import facts_from_label
from hledger_ai.suggest.engine import SuggestionEngine
from hledger_ai.training.state import StateStore
from hledger_ai.training.trainer import TrainingData, train_model

TOTALS = ["11,54", "7,20", "23,05", "4,99", "15,75", "31,10", "8,40", "12,00"]


class FakeOcr:
    """Returns the invented OCR result registered for an image."""

    engine_id = "rapidocr-default"

    def __init__(self) -> None:
        self.by_sha: Dict[str, OcrResult] = {}
        self.calls = 0

    def read(self, *, image_path: str, image_sha256: str) -> OcrResult:
        self.calls += 1
        return self.by_sha[image_sha256]


def _dataset(
    tmp_path: Path, ocr: FakeOcr, *, private: bool = True
) -> TrainingData:
    """Sixteen receipts, alternating card and cash, one per day."""
    examples: List[ReceiptExample] = []
    labels = []
    for index, total in enumerate(TOTALS * 2):
        day = f"2025-03-{index + 1:02d}T10:00:00"
        printed = f"{index + 1:02d}-03-2025 10:00"
        paid = float(total.replace(",", "."))
        if index % 2 == 0:
            label = make_label(date=day, amount=paid, change=0.0)
            evidence = card_receipt(total=total, printed_date=printed)
        else:
            label = make_label(
                date=day,
                amount=50.0,
                change=round(50.0 - paid, 2),
                bank="wallet",
                account_type="physical",
                category="food:market",
                shop_name="Markt De Kraam",
            )
            evidence = cash_receipt(
                total=total,
                given="50,00",
                back=f"{50.0 - paid:.2f}".replace(".", ","),
                printed_date=printed,
            )
        image = tmp_path / "images" / f"receipt_{index}.png"
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4), (index, 0, 0)).save(image)
        sha = file_sha256(path=str(image))
        ocr.by_sha[sha] = OcrResult(
            **{
                **evidence.ocr[0].__dict__,
                "image_sha256": sha,
                "engine": ocr.engine_id,
            }
        )
        labels.append(label)
        examples.append(
            ReceiptExample(
                receipt_id=f"r{index}",
                facts=facts_from_label(label=label),
                image_paths=(str(image),),
                private=private,
                label=label,
            )
        )
    return TrainingData(
        examples=tuple(examples),
        private=private,
        dataset_hash=dataset_hash(labels=labels, private=private),
    )


def _train(tmp_path: Path, ocr: FakeOcr, data: TrainingData, **options):
    return train_model(
        spec=get_model(name="rapidocr-linear"),
        data=data,
        store=StateStore(models_dir=str(tmp_path / "models")),
        cache=OcrCache(cache_dir=str(tmp_path / "cache")),
        ocr_engine=ocr,
        folds=options.pop("folds", 3),
        **options,
    )


def test_training_scores_strategies_on_unseen_receipts_and_saves_an_engine(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)

    state = _train(tmp_path, ocr, data)

    assert state.finished is True
    assert state.private is True
    assert (state.ocr_done, state.ocr_total) == (16, 16)
    assert (state.fits_done, state.fits_total) == (4, 4)  # 3 folds + final
    amount = state.strategies["ST-11"]
    # 16 receipts in 4 chunks: the first is train-only, 12 are tested.
    assert (amount.infer_done, amount.infer_total) == (12, 12)
    assert amount.field_accuracy(field="amount") == 1.0
    engine = SuggestionEngine.load(
        path=StateStore(models_dir=str(tmp_path / "models")).engine_path(
            model="rapidocr-linear", dataset_hash=data.dataset_hash
        )
    )
    guess = engine.suggest(evidence=card_receipt(total="6,35"), known={})
    assert guess["amount"][0].value == 6.35


def test_unchanged_data_is_not_trained_again(tmp_path: Path) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)
    first = _train(tmp_path, ocr, data)
    calls_after_first = ocr.calls

    second = _train(tmp_path, ocr, data)

    assert ocr.calls == calls_after_first == 16
    assert second.updated_at == first.updated_at


def test_force_trains_again_but_never_repeats_ocr(tmp_path: Path) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)
    first = _train(tmp_path, ocr, data)
    first_fit_seconds = first.fit_seconds

    forced = _train(tmp_path, ocr, data, force=True)

    assert ocr.calls == 16
    assert forced.finished is True
    assert forced.fits_done == 4
    assert forced.fit_seconds != first_fit_seconds


def test_an_edited_label_is_a_new_configuration_trained_separately(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)
    _train(tmp_path, ocr, data)
    edited = list(data.examples)
    label = dict(edited[0].label, receipt_category="food:pastry")
    edited[0] = ReceiptExample(
        receipt_id=edited[0].receipt_id,
        facts=facts_from_label(label=label),
        image_paths=edited[0].image_paths,
        private=True,
        label=label,
    )
    changed = TrainingData(
        examples=tuple(edited),
        private=True,
        dataset_hash=dataset_hash(
            labels=[e.label for e in edited], private=True
        ),
    )

    state = _train(tmp_path, ocr, changed)

    store = StateStore(models_dir=str(tmp_path / "models"))
    assert changed.dataset_hash != data.dataset_hash
    assert state.finished is True
    # The older configuration stays (DS-3).
    assert (
        store.load(model="rapidocr-linear", dataset_hash=data.dataset_hash)
        is not None
    )
    assert ocr.calls == 16


def test_training_can_be_limited_to_named_strategies(tmp_path: Path) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)

    state = _train(tmp_path, ocr, data, strategy_ids={"ST-10"})

    assert list(state.strategies) == ["ST-10"]
    assert state.strategies["ST-10"].field_accuracy(field="date") == 1.0


def test_interrupted_training_resumes_without_redoing_finished_folds(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)
    seen: List[int] = []

    def stop_after_first_fold(state) -> None:
        seen.append(state.fits_done)
        if state.fits_done == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _train(tmp_path, ocr, data, on_progress=stop_after_first_fold)
    store = StateStore(models_dir=str(tmp_path / "models"))
    partial = store.load(
        model="rapidocr-linear", dataset_hash=data.dataset_hash
    )
    assert partial.finished is False
    assert list(partial.strategies["ST-11"].folds) == ["0"]

    fits: List[int] = []
    resumed = _train(
        tmp_path, ocr, data, on_progress=lambda s: fits.append(s.fits_done)
    )

    assert resumed.finished is True
    # Fold 0 was kept: only the remaining two folds and the final fit ran.
    assert [f for f in fits if f > 1] and min(f for f in fits if f) >= 1
    assert sorted(resumed.strategies["ST-11"].folds) == ["0", "1", "2"]
    assert resumed.fits_done == 4
    assert ocr.calls == 16


def test_priors_model_trains_without_any_ocr(tmp_path: Path) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr)

    state = train_model(
        spec=get_model(name="priors"),
        data=data,
        store=StateStore(models_dir=str(tmp_path / "models")),
        cache=OcrCache(cache_dir=str(tmp_path / "cache")),
        ocr_engine=None,
        folds=3,
    )

    assert ocr.calls == 0
    assert state.ocr_total == 0
    assert state.finished is True
    assert "ST-11" not in state.strategies
    assert state.strategies["ST-16"].field_accuracy(field="bank_account") == 1.0


def test_public_only_training_is_marked_as_not_private(tmp_path: Path) -> None:
    ocr = FakeOcr()
    data = _dataset(tmp_path, ocr, private=False)

    state = _train(tmp_path, ocr, data)

    assert state.private is False
