"""Orchestrate model training with dataset hash deduplication."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from typeguard import typechecked

from hledger_ai.training.dataset_hash import (
    compute_dataset_hash,
    should_retrain,
)
from hledger_ai.training.export_datasets import (
    export_transaction_dataset,
)

log = logging.getLogger(__name__)


@typechecked
def run_training(
    *,
    labelled_receipts: List[Any],
    models_dir: str,
    feedback_dir: str,
    model_choice: str = "all",
    force_retrain: bool = False,
    ai_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the training workflow.

    1. Export labelled data to JSONL.
    2. Check dataset hash — skip if unchanged (unless force).
    3. Train the requested model(s).
    4. Save the dataset hash into training metadata.

    Parameters
    ----------
    labelled_receipts
        List of Receipt objects from the labels directory.
    models_dir
        Root directory for trained models (e.g. ~/.hledger-ai/models).
    feedback_dir
        Directory containing correction logs.
    model_choice
        "setfit", "qwen-lora", or "all".
    force_retrain
        If True, retrain even if dataset hash matches.
    ai_config
        Optional AIConfig with training parameters.
    """
    models_dir = os.path.expanduser(models_dir)
    feedback_dir = os.path.expanduser(feedback_dir)

    results: Dict[str, Any] = {}

    # --- Step 1: Export transaction dataset ---
    dataset_dir = os.path.join(models_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    tnx_dataset_path = os.path.join(dataset_dir, "transactions.jsonl")

    # Build training examples from labelled receipts.
    labelled_transactions = _extract_training_data(labelled_receipts)

    # Include correction logs if they exist.
    corrections_path = os.path.join(feedback_dir, "corrections.jsonl")
    if os.path.exists(corrections_path):
        labelled_transactions.extend(
            _load_corrections(corrections_path)
        )

    n_exported = export_transaction_dataset(
        labelled_transactions=labelled_transactions,
        output_path=tnx_dataset_path,
    )
    log.info("Exported %d training examples", n_exported)

    if n_exported == 0:
        log.warning("No training data available. Skipping training.")
        return {"error": "No training data", "trained": False}

    # --- Step 2: Train SetFit ---
    if model_choice in ("setfit", "all"):
        setfit_dir = os.path.join(models_dir, "setfit")
        if should_retrain(
            dataset_path=tnx_dataset_path,
            model_dir=setfit_dir,
            force=force_retrain,
        ):
            log.info("Training SetFit classifier...")
            try:
                from hledger_ai.training.train_setfit import (
                    train_setfit_classifier,
                )

                base_model = "sentence-transformers/paraphrase-MiniLM-L6-v2"
                min_examples = 8
                if ai_config is not None:
                    setfit_cfg = ai_config.training.get("setfit", {})
                    base_model = setfit_cfg.get("base_model", base_model)
                    min_examples = setfit_cfg.get(
                        "min_examples_per_class", min_examples
                    )

                setfit_result = train_setfit_classifier(
                    dataset_path=tnx_dataset_path,
                    output_dir=setfit_dir,
                    base_model=base_model,
                    min_examples_per_class=min_examples,
                )
                # Stamp the dataset hash into metadata.
                _stamp_hash(setfit_dir, tnx_dataset_path)
                results["setfit"] = setfit_result
            except ImportError:
                log.warning(
                    "SetFit not installed. "
                    "Install with: pip install setfit sentence-transformers"
                )
                results["setfit"] = {
                    "error": "not installed (pip install setfit sentence-transformers)",
                    "trained": False,
                }
        else:
            results["setfit"] = {"skipped": True, "reason": "dataset unchanged"}

    # --- Step 3: Train Qwen LoRA ---
    if model_choice in ("qwen-lora", "all"):
        lora_dir = os.path.join(models_dir, "qwen-lora")
        if should_retrain(
            dataset_path=tnx_dataset_path,
            model_dir=lora_dir,
            force=force_retrain,
        ):
            log.info("Training Qwen LoRA adapter...")
            try:
                from hledger_ai.training.train_qwen_lora import (
                    train_qwen_lora,
                )

                base_model = "Qwen/Qwen2.5-0.5B"
                if ai_config is not None:
                    lora_cfg = ai_config.training.get("qwen_lora", {})
                    base_model = lora_cfg.get("base_model", base_model)

                lora_result = train_qwen_lora(
                    dataset_path=tnx_dataset_path,
                    output_dir=lora_dir,
                    base_model=base_model,
                )
                _stamp_hash(lora_dir, tnx_dataset_path)
                results["qwen_lora"] = lora_result
            except ImportError:
                log.warning(
                    "Fine-tuning deps not installed. "
                    "Install with: pip install peft trl"
                )
                results["qwen_lora"] = {
                    "error": "not installed (pip install peft trl)",
                    "trained": False,
                }
        else:
            results["qwen_lora"] = {
                "skipped": True,
                "reason": "dataset unchanged",
            }

    return results


def _extract_training_data(
    labelled_receipts: List[Any],
) -> List[Dict[str, str]]:
    """Extract training examples from labelled receipts.

    Each receipt with a category becomes one training example.
    """
    examples: List[Dict[str, str]] = []
    for receipt in labelled_receipts:
        category = getattr(receipt, "receipt_category", None)
        if not category:
            continue

        # Build a description from shop name + items.
        shop = getattr(receipt, "shop_identifier", None)
        shop_name = ""
        if shop is not None:
            shop_name = getattr(shop, "name", "") or ""

        raw_items = getattr(receipt, "net_bought_items", None)
        if raw_items is None:
            items_list = []
        elif isinstance(raw_items, list):
            items_list = raw_items
        else:
            # Single ExchangedItem object.
            items_list = [raw_items]
        item_descs = []
        for item in items_list:
            desc = getattr(item, "description", None)
            if desc:
                item_descs.append(str(desc))

        text_parts = [shop_name] + item_descs
        text = " | ".join(p for p in text_parts if p)

        if text:
            examples.append({"text": text, "label": str(category)})

    return examples


def _load_corrections(
    path: str,
) -> List[Dict[str, str]]:
    """Load correction logs as training examples."""
    examples: List[Dict[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                corrected = obj.get("corrected_category", "")
                text = obj.get("text", "")
                if text and corrected:
                    examples.append({"text": text, "label": corrected})
    except (OSError, json.JSONDecodeError):
        log.warning("Failed to load corrections from %s", path)
    return examples


def _stamp_hash(model_dir: str, dataset_path: str) -> None:
    """Add dataset_hash to the model's training_meta.json."""
    meta_path = Path(model_dir) / "training_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    meta["dataset_hash"] = compute_dataset_hash(dataset_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
