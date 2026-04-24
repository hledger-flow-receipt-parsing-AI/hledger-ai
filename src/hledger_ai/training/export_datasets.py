"""Export labelled data for model training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
def export_transaction_dataset(
    *,
    labelled_transactions: List[Dict[str, Any]],
    output_path: str,
) -> int:
    """Export labelled transactions as JSONL for SetFit training.

    Each entry in *labelled_transactions* should have:
    - ``text``: the classifier input string
      (e.g. ``"ALBERT HEIJN 1543 | AMSTERDAM"``)
    - ``label``: the category path
      (e.g. ``"groceries:albertheijn"``)

    Returns the number of examples written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for entry in labelled_transactions:
            text = entry.get("text", "").strip()
            label = entry.get("label", "").strip()
            if text and label:
                f.write(
                    json.dumps(
                        {"text": text, "label": label},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1

    log.info(
        "Exported %d transaction examples to %s",
        count,
        output_path,
    )
    return count


@typechecked
def export_receipt_dataset(
    *,
    receipt_labels: List[Dict[str, Any]],
    output_path: str,
) -> int:
    """Export receipt label data as JSONL for future VLM fine-tuning.

    Each entry should have:
    - ``image_path``: path to the receipt image
    - ``receipt``: dict of verified receipt fields

    Returns the number of examples written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for entry in receipt_labels:
            img = entry.get("image_path", "")
            receipt = entry.get("receipt")
            if img and receipt:
                f.write(
                    json.dumps(
                        {
                            "image_path": img,
                            "receipt": receipt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1

    log.info(
        "Exported %d receipt examples to %s",
        count,
        output_path,
    )
    return count


@typechecked
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
