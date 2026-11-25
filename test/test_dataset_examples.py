"""Reading labelled, public and unlabelled receipts from a finance root."""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from helpers import make_label
from PIL import Image

from hledger_ai.dataset.examples import (
    load_private_examples,
    load_public_examples,
    unlabelled_image_groups,
)


def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)
    return str(path)


def _write_label(labels: Path, name: str, raw_paths: List[str]) -> None:
    folder = labels / name
    folder.mkdir(parents=True)
    (folder / "receipt_image_to_obj_label.json").write_text(
        json.dumps(make_label(image_paths=raw_paths))
    )


def test_private_examples_pair_each_label_with_its_cropped_photos(
    tmp_path: Path,
) -> None:
    raw = _image(tmp_path / "in" / "20250317_143207.jpg")
    cropped = _image(tmp_path / "proc" / "20250317_143207_cropped.jpg")
    _write_label(tmp_path / "labels", "abc123", [raw, "/gone/missing.jpg"])
    # Left-overs that are not labels must be ignored.
    (tmp_path / "labels" / "empty_folder").mkdir()
    (
        tmp_path / "labels" / "abc123" / "receipt_image_to_obj_label.json.bak"
    ).write_text("{}")

    examples = load_private_examples(
        labels_dir=str(tmp_path / "labels"),
        processed_dir=str(tmp_path / "proc"),
    )

    assert len(examples) == 1
    assert examples[0].receipt_id == "abc123"
    assert examples[0].private is True
    assert examples[0].image_paths == (cropped,)
    assert examples[0].facts.category == "food:bakery"


def test_public_examples_come_from_partial_answers(tmp_path: Path) -> None:
    folder = tmp_path / "20250317_143207"
    cropped = _image(folder / "20250317_143207_cropped.jpg")
    _image(folder / "20250317_143207.jpg")
    (folder / "partial_label.json").write_text(
        json.dumps(
            {
                "complete": False,
                "answers": {
                    "date": "2025-03-17 14:32:00",
                    "is_withdrawal": "n",
                    "category": "food:bakery",
                    "bank_account": "alex:examplebank:checking",
                    "currency": "EUR",
                    "amount": "11.54",
                    "change": "0.0",
                    "add_another": "n",
                },
            }
        )
    )

    examples = load_public_examples(training_dataset_dir=str(tmp_path))

    assert len(examples) == 1
    example = examples[0]
    assert example.private is False
    assert example.image_paths == (cropped,)
    assert example.facts.the_datetime == datetime(2025, 3, 17, 14, 32)
    assert example.facts.category == "food:bakery"
    assert example.facts.amount == 11.54
    assert example.facts.change == 0.0
    assert example.facts.currency == "EUR"
    # Never answered, so unknown rather than invented.
    assert example.facts.shop() is None


def test_missing_public_dataset_folder_means_no_public_examples(
    tmp_path: Path,
) -> None:
    assert (
        load_public_examples(training_dataset_dir=str(tmp_path / "nope")) == []
    )


def test_unlabelled_groups_are_the_groups_without_any_labelled_photo(
    tmp_path: Path,
) -> None:
    a = _image(tmp_path / "in" / "a.jpg")
    b = _image(tmp_path / "in" / "b.jpg")
    c = _image(tmp_path / "in" / "c.jpg")
    d = _image(tmp_path / "in" / "d.jpg")
    e = _image(tmp_path / "in" / "e.jpg")
    grouping = tmp_path / "proc" / "image_grouping.json"
    grouping.parent.mkdir()
    # a+b are one receipt (b carries the label); c+d another; e is new.
    grouping.write_text(json.dumps([[a, b], [c, d]]))
    _write_label(tmp_path / "labels", "label_of_ab", [b])

    groups = unlabelled_image_groups(
        input_dir=str(tmp_path / "in"),
        labels_dir=str(tmp_path / "labels"),
        grouping_path=str(grouping),
    )

    assert groups == [[c, d], [e]]


def test_without_a_grouping_file_every_unlabelled_photo_is_its_own_group(
    tmp_path: Path,
) -> None:
    a = _image(tmp_path / "in" / "a.jpg")
    b = _image(tmp_path / "in" / "b.jpg")
    _write_label(tmp_path / "labels", "label_of_a", [a])

    groups = unlabelled_image_groups(
        input_dir=str(tmp_path / "in"),
        labels_dir=str(tmp_path / "labels"),
        grouping_path=str(tmp_path / "proc" / "image_grouping.json"),
    )

    assert groups == [[b]]


def test_label_folders_without_a_label_are_skipped_silently(
    tmp_path: Path, caplog
) -> None:
    a = _image(tmp_path / "in" / "a.jpg")
    (tmp_path / "labels" / "abandoned_folder").mkdir(parents=True)
    (tmp_path / "labels" / "stray_script.sh").write_text("#!/bin/sh\n")

    groups = unlabelled_image_groups(
        input_dir=str(tmp_path / "in"),
        labels_dir=str(tmp_path / "labels"),
        grouping_path=str(tmp_path / "none.json"),
    )

    assert groups == [[a]]
    assert caplog.records == []
