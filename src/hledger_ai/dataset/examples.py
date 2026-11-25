"""Read receipts from a finance root without changing anything in it.

Private examples are the user's own labelled receipts; public examples are
the receipts moved to ``receipt_training_dataset`` ("This receipt is not
mine"), which carry only the answers typed before they were moved.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typeguard import typechecked

from hledger_ai.receipt_fields import ReceiptFacts, facts_from_label

log = logging.getLogger(__name__)

LABEL_FILENAME = "receipt_image_to_obj_label.json"
PARTIAL_LABEL_FILENAME = "partial_label.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@typechecked
@dataclass(frozen=True)
class ReceiptExample:
    receipt_id: str
    facts: ReceiptFacts
    image_paths: Tuple[str, ...]  # cropped photos that exist, prime first
    private: bool
    label: Optional[Dict[str, Any]] = None  # raw label, for hashing (DS-1)


@typechecked
def cropped_path(
    *,
    raw_path: str,
    processed_dir: str,
    crop_suffix: str = "_cropped",
    crop_ext: str = ".jpg",
) -> str:
    """Where the labeller stores the cropped copy of a raw photo."""
    return os.path.join(
        processed_dir, f"{Path(raw_path).stem}{crop_suffix}{crop_ext}"
    )


@typechecked
def _read_json(*, path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        log.warning("Skipping unreadable file %s", path)
        return None


@typechecked
def load_private_examples(
    *,
    labels_dir: str,
    processed_dir: str,
    label_filename: str = LABEL_FILENAME,
    crop_suffix: str = "_cropped",
    crop_ext: str = ".jpg",
) -> List[ReceiptExample]:
    examples: List[ReceiptExample] = []
    if not os.path.isdir(labels_dir):
        return examples
    for name in sorted(os.listdir(labels_dir)):
        path = os.path.join(labels_dir, name, label_filename)
        if not os.path.isfile(path):
            continue
        label = _read_json(path=path)
        if not isinstance(label, dict):
            continue
        facts = facts_from_label(label=label)
        crops = [
            cropped_path(
                raw_path=raw,
                processed_dir=processed_dir,
                crop_suffix=crop_suffix,
                crop_ext=crop_ext,
            )
            for raw in facts.image_paths
        ]
        examples.append(
            ReceiptExample(
                receipt_id=name,
                facts=facts,
                image_paths=tuple(p for p in crops if os.path.isfile(p)),
                private=True,
                label=label,
            )
        )
    return examples


@typechecked
def _as_float(*, text: Optional[str]) -> Optional[float]:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


@typechecked
def _facts_from_answers(*, answers: Dict[str, Any]) -> ReceiptFacts:
    """Facts from the labeller's partial answers (keyed by question id)."""
    moment: Optional[datetime] = None
    try:
        moment = datetime.fromisoformat(str(answers.get("date", "")))
    except ValueError:
        pass
    return ReceiptFacts(
        the_datetime=moment,
        category=answers.get("category") or None,
        shop_name=None,
        street=None,
        house_nr=None,
        zipcode=None,
        city=None,
        country=None,
        bank_account=answers.get("bank_account") or None,
        currency=answers.get("currency") or None,
        amount=_as_float(text=answers.get("amount")),
        change=_as_float(text=answers.get("change")),
        subtotal=_as_float(text=answers.get("subtotal")),
        total_tax=_as_float(text=answers.get("total_tax")),
        is_withdrawal=answers.get("is_withdrawal") == "y",
        csv_other_party=None,
        csv_description=None,
        image_paths=(),
    )


@typechecked
def load_public_examples(
    *, training_dataset_dir: str, crop_suffix: str = "_cropped"
) -> List[ReceiptExample]:
    examples: List[ReceiptExample] = []
    if not os.path.isdir(training_dataset_dir):
        return examples
    for name in sorted(os.listdir(training_dataset_dir)):
        folder = os.path.join(training_dataset_dir, name)
        data = _read_json(path=os.path.join(folder, PARTIAL_LABEL_FILENAME))
        if not isinstance(data, dict):
            continue
        crops = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if Path(f).stem.endswith(crop_suffix)
            and f.lower().endswith(IMAGE_EXTENSIONS)
        )
        examples.append(
            ReceiptExample(
                receipt_id=name,
                facts=_facts_from_answers(answers=data.get("answers") or {}),
                image_paths=tuple(crops),
                private=False,
                label=data,
            )
        )
    return examples


@typechecked
def _labelled_raw_paths(*, labels_dir: str, label_filename: str) -> set:
    labelled: set = set()
    if not os.path.isdir(labels_dir):
        return labelled
    for name in os.listdir(labels_dir):
        path = os.path.join(labels_dir, name, label_filename)
        label = _read_json(path=path) if os.path.isfile(path) else None
        if isinstance(label, dict):
            labelled.update(facts_from_label(label=label).image_paths)
    return labelled


@typechecked
def unlabelled_image_groups(
    *,
    input_dir: str,
    labels_dir: str,
    grouping_path: str,
    label_filename: str = LABEL_FILENAME,
) -> List[List[str]]:
    """Raw photo groups that still need a label, as the labeller sees them:
    saved groups first, then photos that are not in any group."""
    photos = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    saved = (
        _read_json(path=grouping_path) if os.path.isfile(grouping_path) else []
    )
    groups: List[List[str]] = []
    grouped: set = set()
    for group in saved or []:
        fresh = [p for p in group if p not in grouped]
        grouped.update(fresh)
        if fresh:
            groups.append(fresh)
    groups.extend([p] for p in photos if p not in grouped)
    labelled = _labelled_raw_paths(
        labels_dir=labels_dir, label_filename=label_filename
    )
    return [g for g in groups if not any(p in labelled for p in g)]
