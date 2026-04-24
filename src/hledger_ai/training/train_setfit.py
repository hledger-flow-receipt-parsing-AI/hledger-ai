"""Train a SetFit model for transaction classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
def train_setfit_classifier(
    *,
    dataset_path: str,
    output_dir: str,
    base_model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    num_iterations: int = 20,
    num_epochs: int = 1,
    min_examples_per_class: int = 8,
) -> Dict[str, Any]:
    """Train a SetFit model on labelled transactions.

    Requires ``pip install hledger-ai[setfit]``.

    Parameters
    ----------
    dataset_path
        Path to JSONL file with ``{"text": ..., "label": ...}`` lines.
    output_dir
        Where to save the trained model.
    base_model
        Sentence-transformer base for contrastive learning.
    num_iterations
        Number of text pairs to generate per class.
    num_epochs
        Training epochs for the contrastive step.
    min_examples_per_class
        Skip training if any class has fewer than this.

    Returns
    -------
    dict
        Metrics: accuracy, f1, num_classes, num_examples.
    """
    try:
        from datasets import Dataset
        from setfit import SetFitModel, SetFitTrainer
    except ImportError:
        raise ImportError(
            "SetFit dependencies not installed. "
            "Install with: pip install setfit sentence-transformers"
        )

    # Load data
    examples = _load_dataset(dataset_path)
    if not examples:
        return {"error": "No examples found", "trained": False}

    # Check minimum examples per class
    label_counts = _count_labels(examples)
    too_few = {
        k: v
        for k, v in label_counts.items()
        if v < min_examples_per_class
    }
    if too_few:
        log.warning(
            "Classes with too few examples (need %d): %s",
            min_examples_per_class,
            too_few,
        )

    valid_labels = {
        k
        for k, v in label_counts.items()
        if v >= min_examples_per_class
    }
    if len(valid_labels) < 2:
        return {
            "error": "Need at least 2 classes with "
            f">={min_examples_per_class} examples",
            "trained": False,
            "label_counts": label_counts,
        }

    # Filter to valid labels
    filtered = [
        e for e in examples if e["label"] in valid_labels
    ]
    log.info(
        "Training on %d examples across %d classes",
        len(filtered),
        len(valid_labels),
    )

    # Split train/eval (80/20 stratified)
    train_data, eval_data = _stratified_split(
        filtered, test_ratio=0.2
    )

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    # Create model
    model = SetFitModel.from_pretrained(base_model)

    # Train
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
    )
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    log.info("Evaluation metrics: %s", metrics)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    # Save metadata
    meta = {
        "base_model": base_model,
        "num_examples": len(filtered),
        "num_classes": len(valid_labels),
        "classes": sorted(valid_labels),
        "metrics": metrics,
        "label_counts": label_counts,
    }
    with open(out / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Model saved to %s", output_dir)
    return {
        "trained": True,
        "output_dir": output_dir,
        **meta,
    }


def _load_dataset(path: str) -> List[Dict[str, str]]:
    """Load JSONL dataset."""
    entries: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "text" in obj and "label" in obj:
                    entries.append(obj)
    return entries


def _count_labels(
    examples: List[Dict[str, str]],
) -> Dict[str, int]:
    """Count examples per label."""
    counts: Dict[str, int] = {}
    for ex in examples:
        label = ex["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _stratified_split(
    examples: List[Dict[str, str]],
    *,
    test_ratio: float = 0.2,
) -> tuple:
    """Split into train/test maintaining class ratios."""
    import random

    by_label: Dict[str, List[Dict[str, str]]] = {}
    for ex in examples:
        by_label.setdefault(ex["label"], []).append(ex)

    train: List[Dict[str, str]] = []
    test: List[Dict[str, str]] = []

    for label, items in by_label.items():
        shuffled = list(items)
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * test_ratio))
        test.extend(shuffled[:split_idx])
        train.extend(shuffled[split_idx:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test
