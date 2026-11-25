"""The suggestion service: trained models + policy, as the labeller uses it."""

from pathlib import Path

from helpers import card_receipt
from test_app_training import FakeOcr, _ai, _finance_root

from hledger_ai.app import train
from hledger_ai.suggest.service import SuggestionService


def _trained(tmp_path: Path):
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )
    return finance, ai


def test_each_field_is_answered_by_its_most_accurate_model(
    tmp_path: Path,
) -> None:
    _, ai = _trained(tmp_path)
    service = SuggestionService.load(ai=ai)

    suggestions = service.suggest(
        evidence={"rapidocr-default": card_receipt(total="6,35")}, known={}
    )

    # Only the image model can read an amount.
    assert suggestions["amount"][0].value == 6.35
    assert suggestions["amount"][0].sources == ("ST-11",)
    # Every training receipt is in EUR: the image model cannot beat "always
    # EUR", so the cheaper prior answers.
    assert suggestions["currency"][0].value == "EUR"
    assert suggestions["currency"][0].sources == ("ST-19",)


def test_without_a_photo_the_priors_still_answer(tmp_path: Path) -> None:
    _, ai = _trained(tmp_path)
    service = SuggestionService.load(ai=ai)

    suggestions = service.suggest(evidence={}, known={})

    assert suggestions["currency"][0].value == "EUR"
    assert "amount" not in suggestions


def test_confirmed_fields_are_left_alone(tmp_path: Path) -> None:
    _, ai = _trained(tmp_path)
    service = SuggestionService.load(ai=ai)

    suggestions = service.suggest(
        evidence={"rapidocr-default": card_receipt(total="6,35")},
        known={"amount": 9.99, "currency": "USD"},
    )

    assert "amount" not in suggestions
    assert "currency" not in suggestions


def test_service_without_trained_models_suggests_nothing(
    tmp_path: Path,
) -> None:
    service = SuggestionService.load(ai=_ai(tmp_path))

    assert service.suggest(evidence={}, known={}) == {}
    assert service.ocr_engine_ids() == []


def test_service_names_the_ocr_engines_its_models_need(tmp_path: Path) -> None:
    _, ai = _trained(tmp_path)

    assert SuggestionService.load(ai=ai).ocr_engine_ids() == [
        "rapidocr-default"
    ]


def test_newest_trained_configuration_is_the_one_in_use(tmp_path: Path) -> None:
    import json

    finance, ai = _trained(tmp_path)
    first = SuggestionService.load(ai=ai).dataset_hashes()["priors"]
    # Relabel one receipt: a new dataset configuration.
    label_path = Path(finance.labels_dir) / "label0" / finance.label_filename
    label = json.loads(label_path.read_text())
    label["receipt_category"] = "food:pastry"
    label_path.write_text(json.dumps(label))
    train(finance=finance, ai=ai, names=["priors"], folds=2)

    second = SuggestionService.load(ai=ai).dataset_hashes()["priors"]

    assert second != first


def test_later_public_only_training_does_not_replace_the_private_models(
    tmp_path: Path,
) -> None:
    # The public dataset is tiny; labelling your own receipts should keep
    # using the models trained on all your labels.
    finance, ai = _trained(tmp_path)
    private_hash = SuggestionService.load(ai=ai).dataset_hashes()["priors"]

    train(finance=finance, ai=ai, names=["priors"], folds=1, public_only=True)

    assert (
        SuggestionService.load(ai=ai).dataset_hashes()["priors"] == private_hash
    )


def test_public_models_are_used_when_nothing_else_is_trained(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)

    states = train(
        finance=finance, ai=ai, names=["priors"], folds=1, public_only=True
    )

    assert SuggestionService.load(ai=ai).dataset_hashes() == {
        "priors": states[0].dataset_hash
    }
