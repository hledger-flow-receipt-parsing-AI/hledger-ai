"""The adapter between the suggestion service and the labelling TUI."""

from datetime import date, datetime
from pathlib import Path

from hledger_config.config.AIConfig import AIConfig
from hledger_config.config.CategorisationConfig import CategorisationConfig
from hledger_config.config.Config import Config
from hledger_config.config.DirPathsConfig import DirPathsConfig
from hledger_config.config.Filenames import FileNames
from hledger_config.config.MatchingAlgoConfig import MatchingAlgoConfig
from hledger_config.config.ReceiptImgConfig import ReceiptImgConfig
from helpers import card_receipt, make_evidence, make_label
from PIL import Image
from test_app_training import FakeOcr, _ai, _finance_root

from hledger_ai.app import train
from hledger_ai.labeller import (
    LabellerSuggester,
    known_from_answers,
    to_labeller_suggestions,
)
from hledger_ai.ocr.cache import file_sha256
from hledger_ai.ocr.types import OcrResult
from hledger_ai.suggest.engine import Suggestion


def test_answers_become_known_fields() -> None:
    known = known_from_answers(
        answers={
            "date": datetime(2025, 3, 17, 14, 32),
            "is_withdrawal": "n",
            "category": "food:bakery",
            "bank_account": "alex:examplebank:checking",
            "currency": "EUR",
            "amount": "11.54",
            "change": 0.0,
            "address_selector": "Bakker Bart: Dorpsstraat, 1",
            "subtotal": "",
            "add_another": "n",
        }
    )

    assert known == {
        "date": date(2025, 3, 17),
        "time": "14:32",
        "is_withdrawal": False,
        "category": "food:bakery",
        "bank_account": "alex:examplebank:checking",
        "currency": "EUR",
        "amount": 11.54,
        "change": 0.0,
        "shop": "Bakker Bart: Dorpsstraat, 1",
    }


def test_typed_amounts_read_the_same_with_a_decimal_comma_or_spaces() -> None:
    known = known_from_answers(
        answers={"amount": " 6,35 ", "change": "0,00", "subtotal": "6.35 "}
    )

    assert known == {"amount": 6.35, "change": 0.0, "subtotal": 6.35}


def test_typed_text_answers_lose_surrounding_spaces() -> None:
    known = known_from_answers(
        answers={"category": " food:bakery ", "currency": "EUR "}
    )

    assert known == {"category": "food:bakery", "currency": "EUR"}


def test_manual_address_entry_is_not_a_known_shop() -> None:
    known = known_from_answers(
        answers={
            "address_selector": "manual address",
            "shop_name": "Nieuwe Winkel",
            "shop_zipcode": "1234 AB",
        }
    )

    assert known == {"shop_name": "Nieuwe Winkel", "address": "1234 AB"}


def test_suggestions_are_keyed_and_written_the_way_the_questions_expect() -> (
    None
):
    converted = to_labeller_suggestions(
        suggestions={
            "date": [
                Suggestion(date(2025, 3, 17), 0.9, ("ST-10",)),
                Suggestion(date(2025, 3, 31), 0.1, ("ST-10",)),
            ],
            "time": [Suggestion("14:32", 0.8, ("ST-10",))],
            "amount": [Suggestion(11.5, 0.7, ("ST-11",))],
            "change": [Suggestion(0.0, 0.6, ("ST-11", "ST-17"))],
            "shop": [
                Suggestion("Bakker Bart: Dorpsstraat, 1", 0.5, ("ST-12",))
            ],
            "is_withdrawal": [Suggestion("n", 0.97, ("ST-19",))],
        }
    )

    assert [s.question for s in converted["receipt_date"]] == [
        "2025-03-17 14:32",
        "2025-03-31 14:32",
    ]
    assert converted["receipt_date"][0].probability == 0.9
    assert converted["amount"][0].question == "11.50"
    assert converted["change"][0].question == "0.00"
    assert converted["change"][0].model_name == "ST-11+ST-17"
    assert converted["address_selector"][0].question == (
        "Bakker Bart: Dorpsstraat, 1"
    )
    assert converted["is_withdrawal"][0].question == "n"


def test_date_without_a_readable_time_is_suggested_at_midnight() -> None:
    converted = to_labeller_suggestions(
        suggestions={"date": [Suggestion(date(2025, 3, 17), 0.9, ("ST-10",))]}
    )

    assert converted["receipt_date"][0].question == "2025-03-17 00:00"


def _config(tmp_path: Path) -> Config:
    root = tmp_path / "finance"
    (root / "categories.yaml").write_text("food: {}\n", encoding="utf-8")
    return Config(
        accounts=[],
        dir_paths=DirPathsConfig(
            root_finance_path=str(root),
            working_subdir="work",
            receipt_images_input_dir="in",
            hledger_plot_dir="plots",
            receipt_images_processed_dir="proc",
            receipt_labels_dir="labels",
            asset_transaction_csvs_dir="asset_transaction_csvs",
        ),
        file_names=FileNames(
            start_journal_filepath="start.journal",
            root_journal_filename="all.journal",
            tui_label_filename="receipt_image_to_obj_label",
            receipt_img=ReceiptImgConfig(
                processing_metadata_ext=".json",
                rotate="_rotated",
                rotate_ext=".jpg",
                crop="_cropped",
                crop_ext=".jpg",
            ),
            categories_filename="categories.yaml",
        ),
        categorisation=CategorisationConfig(quick=False),
        matching_algo=MatchingAlgoConfig(
            days=2,
            amount_range=0.0,
            days_month_swap=False,
            multiple_receipts_per_transaction=False,
        ),
        csv_encoding="utf-8",
        ai=AIConfig(
            models_dir=str(tmp_path / "ai" / "models"),
            feedback_dir=str(tmp_path / "ai" / "feedback"),
        ),
    )


def _new_receipt(tmp_path: Path, ocr: FakeOcr) -> str:
    """An unlabelled photo with its cropped copy; returns the raw path."""
    raw = tmp_path / "finance" / "in" / "20250401_090000.jpg"
    crop = tmp_path / "finance" / "proc" / "20250401_090000_cropped.jpg"
    for path in (raw, crop):
        Image.new("RGB", (4, 4), (200, 200, 0)).save(path)
    evidence = card_receipt(total="6,35", printed_date="01-04-2025 09:00")
    sha = file_sha256(path=str(crop))
    ocr.by_sha[sha] = OcrResult(
        **{
            **evidence.ocr[0].__dict__,
            "image_sha256": sha,
            "engine": ocr.engine_id,
        }
    )
    return str(raw)


def test_labeller_gets_suggestions_for_a_new_receipt(tmp_path: Path) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )
    raw = _new_receipt(tmp_path, ocr)

    suggester = LabellerSuggester.open(
        config=_config(tmp_path),
        raw_image_paths=[raw],
        ocr_engines={"rapidocr-default": ocr},
    )
    first = suggester.suggest(answers={})
    later = suggester.suggest(answers={"amount": 6.35, "currency": "EUR"})

    assert first["receipt_date"][0].question == "2025-04-01 09:00"
    assert first["amount"][0].question == "6.35"
    assert first["currency"][0].question == "EUR"
    assert "amount" not in later and "currency" not in later
    assert "receipt_date" in later


def test_no_trained_model_means_no_suggester(tmp_path: Path) -> None:
    ocr = FakeOcr()
    _finance_root(tmp_path, ocr)
    raw = _new_receipt(tmp_path, ocr)

    assert (
        LabellerSuggester.open(
            config=_config(tmp_path),
            raw_image_paths=[raw],
            ocr_engines={"rapidocr-default": ocr},
        )
        is None
    )


def test_field_events_are_timed_through_the_suggester(tmp_path: Path) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(finance=finance, ai=ai, names=["priors"], folds=1)
    raw = _new_receipt(tmp_path, ocr)
    suggester = LabellerSuggester.open(
        config=_config(tmp_path), raw_image_paths=[raw], ocr_engines={}
    )

    suggester.reached(question_id="category")
    suggester.answered(question_id="category", answer="food:bakery")

    lines = Path(ai.timings_path).read_text().splitlines()
    assert len(lines) == 2
    assert '"event": "reached"' in lines[0]
    assert '"event": "entered"' in lines[1]
    assert "food:bakery" not in "".join(lines)


def test_finishing_logs_only_the_suggestions_the_user_did_not_keep(
    tmp_path: Path,
) -> None:
    import json

    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )
    raw = _new_receipt(tmp_path, ocr)
    suggester = LabellerSuggester.open(
        config=_config(tmp_path),
        raw_image_paths=[raw],
        ocr_engines={"rapidocr-default": ocr},
    )
    suggester.suggest(answers={})  # first guesses: amount 6.35, EUR, ...

    logged = suggester.finish(
        answers={
            "date": "2025-04-01 09:00:00",  # kept (seconds do not matter)
            "amount": "6.35",  # kept
            "currency": "USD",  # corrected
        }
    )

    entries = [
        json.loads(line)
        for line in Path(ai.corrections_path).read_text().splitlines()
    ]
    assert logged == 1
    assert [(e["field"], e["suggested"], e["accepted"]) for e in entries] == [
        ("currency", "EUR", "USD")
    ]
    assert entries[0]["image_path"] == raw


def test_finishing_without_corrections_writes_nothing(tmp_path: Path) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(finance=finance, ai=ai, names=["priors"], folds=1)
    raw = _new_receipt(tmp_path, ocr)
    suggester = LabellerSuggester.open(
        config=_config(tmp_path), raw_image_paths=[raw], ocr_engines={}
    )
    suggester.suggest(answers={})

    assert suggester.finish(answers={"currency": "EUR"}) == 0
    assert not Path(ai.corrections_path).exists()


def test_highlights_point_at_the_boxes_behind_the_suggestions(
    tmp_path: Path,
) -> None:
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )
    raw = _new_receipt(tmp_path, ocr)
    suggester = LabellerSuggester.open(
        config=_config(tmp_path),
        raw_image_paths=[raw],
        ocr_engines={"rapidocr-default": ocr},
    )
    assert suggester.highlights() == {}  # nothing suggested yet

    suggester.suggest(answers={})
    highlights = suggester.highlights()

    crop = str(tmp_path / "finance" / "proc" / "20250401_090000_cropped.jpg")
    assert [h.text for h in highlights["amount"]] == ["6,35", "6,35"]
    assert {h.image_path for h in highlights["amount"]} == {crop}
    # The date question gets the date and the time; here one box has both.
    assert [h.text for h in highlights["date"]] == ["Datum 01-04-2025 09:00"]
    # The category is told from the printed text, so its boxes are text
    # lines; every label has the same payment account, so nothing tells
    # that one apart and nothing is shown for it.
    assert "6,35" not in [h.text for h in highlights["category"]]
    assert "bank_account" not in highlights


def test_leaving_the_subtotal_empty_is_suggested_as_an_empty_answer() -> None:
    converted = to_labeller_suggestions(
        suggestions={
            "subtotal": [
                Suggestion(None, 0.8, ("ST-15",)),
                Suggestion(12.5, 0.2, ("ST-15",)),
            ]
        }
    )

    assert [s.question for s in converted["subtotal"]] == ["", "12.50"]


def _add_subtotal_receipts(tmp_path: Path, ocr: FakeOcr) -> None:
    """Supermarket receipts that print a subtotal, labelled with it, on the
    same days as the fixture's receipts without one."""
    import json

    root = tmp_path / "finance"
    for index in range(12):
        paid = 5.0 + 1.5 * index
        before = paid + 1.25
        raw = root / "in" / f"2025030{index % 9 + 1}_11000{index}.jpg"
        crop = root / "proc" / f"{raw.stem}_cropped.jpg"
        for path in (raw, crop):
            Image.new("RGB", (4, 4), (index, 3, 0)).save(path)
        evidence = make_evidence(
            rows=[
                ("Supermarkt Zonneveld",),
                (f"Datum {index + 1:02d}-03-2025 11:{index:02d}",),
                ("Kaas", "4,10"),
                ("SUBTOTAAL", f"{before:.2f}".replace(".", ",")),
                ("Korting", "1,25"),
                ("TOTAAL", f"{paid:.2f}".replace(".", ",")),
                ("PIN", f"{paid:.2f}".replace(".", ",")),
            ]
        )
        sha = file_sha256(path=str(crop))
        ocr.by_sha[sha] = OcrResult(
            **{
                **evidence.ocr[0].__dict__,
                "image_sha256": sha,
                "engine": ocr.engine_id,
            }
        )
        folder = root / "labels" / f"supermarket{index}"
        folder.mkdir(parents=True)
        (folder / "receipt_image_to_obj_label.json").write_text(
            json.dumps(
                make_label(
                    date=f"2025-03-{index + 1:02d}T11:{index:02d}:00",
                    shop_name="Supermarkt Zonneveld",
                    amount=paid,
                    subtotal=before,
                    image_paths=[str(raw)],
                )
            )
        )


def test_an_empty_subtotal_counts_as_an_answer_when_finishing(
    tmp_path: Path,
) -> None:
    import json

    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    _add_subtotal_receipts(tmp_path, ocr)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )

    def finish(answers: dict) -> list:
        suggester = LabellerSuggester.open(
            config=_config(tmp_path),
            raw_image_paths=[_new_receipt(tmp_path, ocr)],
            ocr_engines={"rapidocr-default": ocr},
        )
        first = suggester.suggest(answers={})
        # The new receipt prints no subtotal: leave it empty.
        assert first["subtotal"][0].question == ""
        suggester.finish(answers=answers)
        path = Path(ai.corrections_path)
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.unlink()
        return [(r["field"], r["suggested"], r["accepted"]) for r in rows]

    assert finish({"subtotal": ""}) == []  # kept: nothing to log
    assert finish({"subtotal": "4.20"}) == [("subtotal", "", "4.2")]


def test_manual_address_questions_get_suggestions(tmp_path: Path) -> None:
    """A shop entered by hand: its parts are read from the receipt."""
    ocr = FakeOcr()
    finance, ai = _finance_root(tmp_path, ocr), _ai(tmp_path)
    train(
        finance=finance,
        ai=ai,
        names=["priors", "rapidocr-linear"],
        folds=2,
        ocr_engines={"rapidocr-default": ocr},
    )
    raw = _new_receipt(tmp_path, ocr)
    suggester = LabellerSuggester.open(
        config=_config(tmp_path),
        raw_image_paths=[raw],
        ocr_engines={"rapidocr-default": ocr},
    )

    first = suggester.suggest(answers={})

    assert first["shop_name"][0].question == "Voorbeeld Markt"
    assert first["shop_street"][0].question == "Dorpsstraat"
    assert first["shop_house_nr"][0].question == "12"
    assert first["shop_city"][0].question == "Voorbeeldstad"
    assert first["shop_country"][0].question == "Netherlands"
    assert first["shop_street"][0].model_name == "ST-12"  # read

    boxes = suggester.highlights()
    assert [h.text for h in boxes["shop_street"]] == ["Dorpsstraat 12"]
    assert [h.text for h in boxes["shop_name"]] == ["Voorbeeld Markt"]

    later = suggester.suggest(
        answers={
            "address_selector": "manual address",
            "shop_street": "Molenweg",
        }
    )
    assert "shop_street" not in later  # never suggest what was typed
    assert "shop_name" in later
