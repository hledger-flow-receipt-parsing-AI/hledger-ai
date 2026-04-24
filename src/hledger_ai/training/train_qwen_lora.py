"""LoRA fine-tune Qwen3-0.6B for transaction classification.

Works on both CPU and GPU.  GPU is faster but CPU is fully supported.
Requires: ``pip install peft trl``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from typeguard import typechecked

log = logging.getLogger(__name__)

# Default base model — the full-precision HuggingFace version (works on
# CPU and GPU).  Users with a GPU + bitsandbytes can override via config
# to use a quantized variant.
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B"


@typechecked
def train_qwen_lora(
    *,
    dataset_path: str,
    output_dir: str,
    base_model: str = _DEFAULT_BASE_MODEL,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_steps: int = 200,
    learning_rate: float = 2e-4,
) -> Dict[str, Any]:
    """LoRA fine-tune a small language model on labelled transactions.

    Parameters
    ----------
    dataset_path
        JSONL with ``{"text": ..., "label": ...}`` lines.
    output_dir
        Where to save the LoRA adapter.
    base_model
        HuggingFace model ID.  Default ``Qwen/Qwen2.5-0.5B`` works on
        CPU (float32) and GPU (float16/bf16).
    lora_r
        LoRA rank.
    lora_alpha
        LoRA alpha scaling.
    max_steps
        Maximum training steps.
    learning_rate
        Learning rate for LoRA fine-tuning.

    Returns
    -------
    dict
        Training result with output_dir and metadata.
    """
    try:
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Fine-tuning dependencies not installed. "
            "Install with: pip install peft trl"
        )

    # Load dataset
    examples = _load_sft_dataset(dataset_path)
    if len(examples) < 50:
        return {
            "error": f"Only {len(examples)} examples; "
            "need at least 50 for LoRA fine-tuning",
            "trained": False,
        }

    log.info("Fine-tuning on %d examples", len(examples))

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        dtype = torch.float32
        log.info(
            "No GPU detected — training on CPU (this will be slow but works)"
        )

    # Load model + tokenizer
    log.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "LoRA parameters: %d trainable / %d total (%.1f%%)",
        trainable,
        total,
        100.0 * trainable / total,
    )

    # Format training data
    from datasets import Dataset

    def format_example(ex: Dict[str, str]) -> Dict[str, str]:
        return {
            "text": (
                f"Classify this transaction: {ex['text']}\n"
                f"Category: {ex['label']}"
            )
        }

    train_data = [format_example(e) for e in examples]
    dataset = Dataset.from_list(train_data)

    # Train
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(out),
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=2 if device == "cpu" else 4,
        logging_steps=10,
        save_steps=max_steps,
        fp16=(device == "cuda" and dtype == torch.float16),
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()

    # Save LoRA adapter only (not the full model)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Save metadata
    meta = {
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_examples": len(examples),
        "max_steps": max_steps,
        "device": device,
    }
    with open(out / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("LoRA adapter saved to %s", output_dir)
    return {"trained": True, "output_dir": output_dir, **meta}


def _load_sft_dataset(
    path: str,
) -> List[Dict[str, str]]:
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
