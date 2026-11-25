"""Training state: progress, ETA and accuracy per model and strategy."""

from pathlib import Path

import pytest

from hledger_ai.training.state import ModelState, StateStore, StrategyState


def _state(**changes) -> ModelState:
    values = dict(
        model="rapidocr-linear",
        dataset_hash="abc",
        private=True,
        model_version="2",
        hyperparameters={"folds": 4},
    )
    values.update(changes)
    return ModelState(**values)


def test_training_progress_weighs_ocr_and_fitting_by_the_time_they_take() -> (
    None
):
    state = _state(
        ocr_done=50,
        ocr_total=100,
        ocr_read=50,
        ocr_seconds=100.0,  # 2 s per photo
        fits_done=1,
        fits_total=5,
        fit_seconds=10.0,  # 10 s per fit
    )

    # Done: 110 s. Left: 50 photos x 2 s + 4 fits x 10 s = 140 s.
    assert state.training_fraction() == pytest.approx(110 / 250)
    assert state.training_eta_seconds() == pytest.approx(140.0)


def test_training_that_has_not_started_has_no_eta_yet() -> None:
    state = _state(ocr_total=100, fits_total=5)

    assert state.training_fraction() == 0.0
    assert state.training_eta_seconds() is None


def test_finished_training_is_complete_with_nothing_left() -> None:
    state = _state(
        ocr_done=100,
        ocr_total=100,
        ocr_seconds=200.0,
        fits_done=5,
        fits_total=5,
        fit_seconds=50.0,
        finished=True,
    )

    assert state.training_fraction() == 1.0
    assert state.training_eta_seconds() == 0.0


def test_inference_progress_and_eta_follow_the_receipts_processed() -> None:
    strategy = StrategyState(infer_done=30, infer_total=120, infer_seconds=60.0)

    assert strategy.inference_fraction() == pytest.approx(0.25)
    assert strategy.inference_eta_seconds() == pytest.approx(180.0)


def test_accuracy_is_the_average_over_the_improved_fields() -> None:
    strategy = StrategyState(
        folds={
            "0": {
                "date": {"correct": 4, "total": 5, "baseline_correct": 0},
                "time": {"correct": 2, "total": 5, "baseline_correct": 0},
            },
            "1": {
                "date": {"correct": 5, "total": 5, "baseline_correct": 1},
                "time": {"correct": 3, "total": 5, "baseline_correct": 0},
            },
        }
    )

    # date 9/10, time 5/10.
    assert strategy.accuracy() == pytest.approx(0.7)
    assert strategy.baseline_accuracy() == pytest.approx(0.05)
    assert strategy.field_accuracy(field="date") == pytest.approx(0.9)


def test_strategy_without_scored_receipts_has_no_accuracy() -> None:
    assert StrategyState().accuracy() is None


def test_state_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    store = StateStore(models_dir=str(tmp_path))
    state = _state(ocr_done=3, ocr_total=4, ocr_seconds=6.0)
    state.strategies["ST-10"] = StrategyState(
        infer_done=2,
        infer_total=4,
        infer_seconds=1.0,
        folds={
            "0": {"date": {"correct": 2, "total": 2, "baseline_correct": 0}}
        },
    )

    store.save(state=state)
    restored = store.load(model="rapidocr-linear", dataset_hash="abc")

    assert restored == state


def test_missing_state_loads_as_nothing(tmp_path: Path) -> None:
    store = StateStore(models_dir=str(tmp_path))

    assert store.load(model="rapidocr-linear", dataset_hash="abc") is None


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_hash": "other"},
        {"model_version": "3"},
        {"hyperparameters": {"folds": 5}},
    ],
)
def test_model_is_out_of_date_when_data_version_or_settings_differ(
    change: dict,
) -> None:
    state = _state(finished=True)
    current = dict(
        dataset_hash="abc", model_version="2", hyperparameters={"folds": 4}
    )

    assert state.is_current(**current) is True
    current.update(change)
    assert state.is_current(**current) is False


def test_unfinished_training_is_not_current() -> None:
    state = _state(finished=False)

    assert (
        state.is_current(
            dataset_hash="abc", model_version="2", hyperparameters={"folds": 4}
        )
        is False
    )


def test_photos_served_from_the_cache_do_not_make_the_eta_optimistic() -> None:
    # 40 photos were cached, 10 were read at 3 s each; 50 are still to read.
    state = _state(
        ocr_done=50, ocr_total=100, ocr_read=10, ocr_seconds=30.0, fits_total=0
    )

    assert state.training_eta_seconds() == pytest.approx(150.0)
