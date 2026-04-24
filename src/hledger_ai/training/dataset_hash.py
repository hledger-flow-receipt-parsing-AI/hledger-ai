"""Dataset hash computation for training deduplication.

Computes a SHA-256 hash of the sorted, deduplicated dataset so that
training is skipped when the dataset hasn't changed since the last run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
def compute_dataset_hash(dataset_path: str) -> str:
    """Compute a deterministic SHA-256 hash of a JSONL dataset.

    Lines are sorted and deduplicated so that row ordering does not
    affect the hash.
    """
    lines: list[str] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    unique_sorted = sorted(set(lines))
    content = "\n".join(unique_sorted)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@typechecked
def read_stored_hash(model_dir: str) -> Optional[str]:
    """Read the dataset hash from a previous training run."""
    meta_path = Path(model_dir) / "training_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("dataset_hash")
    except (json.JSONDecodeError, OSError):
        return None


@typechecked
def should_retrain(
    *,
    dataset_path: str,
    model_dir: str,
    force: bool = False,
) -> bool:
    """Check whether retraining is needed.

    Returns True if:
    - ``force`` is True
    - No previous model exists
    - The dataset hash has changed since the last training
    """
    if force:
        log.info("Force retrain requested")
        return True

    model_path = Path(model_dir)
    if not model_path.exists():
        log.info("No existing model at %s; training needed", model_dir)
        return True

    current_hash = compute_dataset_hash(dataset_path)
    stored_hash = read_stored_hash(model_dir)

    if stored_hash is None:
        log.info("No stored dataset hash; training needed")
        return True

    if current_hash != stored_hash:
        log.info(
            "Dataset hash changed (%s -> %s); retraining",
            stored_hash[:12],
            current_hash[:12],
        )
        return True

    log.info("Dataset unchanged (hash %s); skipping training", current_hash[:12])
    return False
